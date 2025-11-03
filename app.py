from flask import Flask, render_template, Response, jsonify
import cv2, dlib, numpy as np, time, threading, pygame
from scipy.spatial import distance

app = Flask(__name__)

# Load Dlib face detector & shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize pygame mixer for sounds
pygame.mixer.init()
praise_sound = "static/assets/praise.mp3"
scold_sound = "static/assets/scold.mp3"

# Global states
attention_score = 100
blink_count = 0
yawn_count = 0

# Thresholds
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.65
LOOK_AWAY_THRESH = 35  # degrees

# ------------------------- Utility Functions -------------------------

def calculate_EAR(eye):
    """Eye Aspect Ratio (EAR) - measures blink/eye closure."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_MAR(mouth):
    """Mouth Aspect Ratio (MAR) - measures yawning."""
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


def get_head_pose(landmarks, frame_shape):
    """Estimate head rotation (left/right)."""
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    height, width = frame_shape
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    yaw = euler_angles[1][0]
    return yaw


def play_sound(file):
    """Play alert sound safely."""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

# ------------------------- Core Logic -------------------------

def gen_frames():
    """Generate live webcam frames."""
    global attention_score, blink_count, yawn_count
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        score = 100

        for face in faces:
            landmarks = predictor(gray, face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye = points[36:42]
            right_eye = points[42:48]
            mouth = points[48:68]

            ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
            mar = calculate_MAR(mouth)
            yaw = get_head_pose(landmarks, frame.shape[:2])

            if ear < EYE_AR_THRESH:
                blink_count += 1
                score -= 15

            if mar > MOUTH_AR_THRESH:
                yawn_count += 1
                score -= 20

            if abs(yaw) > LOOK_AWAY_THRESH:
                score -= 25

        attention_score = int(0.8 * attention_score + 0.2 * max(score, 0))

        if attention_score > 85:
            feedback = "Fully Focused 🔥"
            color = (0, 255, 0)
        elif attention_score > 60:
            feedback = "Slightly Distracted 😐"
            color = (0, 255, 255)
        else:
            feedback = "Drowsy / Unfocused 😴"
            color = (0, 0, 255)
            threading.Thread(target=play_sound, args=(scold_sound,), daemon=True).start()

        cv2.putText(frame, f"Attention: {attention_score}%", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, feedback, (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ------------------------- Flask Routes -------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_score')
def get_score():
    return jsonify({'score': int(attention_score)})

# ------------------------- Run App -------------------------
if __name__ == "__main__":
    app.run(debug=True)
