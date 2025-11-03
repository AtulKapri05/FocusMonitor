import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance
import pyttsx3

# ------------------------------------------------------
# 🔊 Text-to-speech setup  (initialize once)
# ------------------------------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------------------------------------------
# 🎯 Eye & Mouth Aspect Ratios
# ------------------------------------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth):
    # use 12 points (48-59)
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# ------------------------------------------------------
# 📸  Dlib setup
# ------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ------------------------------------------------------
# 🎥  Webcam
# ------------------------------------------------------
cap = cv2.VideoCapture(0)
prev_time = 0
total_frames = 0
attention_sum = 0

# Global score for Flask dashboard
global_attention_score = 0

# ------------------------------------------------------
# ⚙️  Head pose model points
# ------------------------------------------------------
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye corner
    (225.0, 170.0, -135.0),     # Right eye corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
])

# ------------------------------------------------------
# 🔁  Main loop
# ------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    attention_score = 100

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])

        left_eye = points[36:42]
        right_eye = points[42:48]
        mouth = points[48:60]

        leftEAR = calculate_EAR(left_eye)
        rightEAR = calculate_EAR(right_eye)
        EAR = (leftEAR + rightEAR) / 2.0
        MAR = calculate_MAR(mouth)

        # 👁️ Blink
        if EAR < 0.22:
            attention_score -= 10
            cv2.putText(frame, "Blink Detected!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 😴 Yawn
        if MAR > 0.6:
            attention_score -= 15
            cv2.putText(frame, "Yawn Detected!", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 🧭 Head pose
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
        ], dtype="double")

        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        nose_end, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        if abs(p2[0] - p1[0]) > 100:
            attention_score -= 20
            cv2.putText(frame, "Looking Away!", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        total_frames += 1
        attention_sum += max(attention_score, 0)

    # 🕒 FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Attention: {max(attention_score, 0)}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 📊  Attention bar
    bar_x, bar_y = 30, 100
    bar_width = 300
    bar_height = 20
    score = max(min(attention_score, 100), 0)
    fill = int((score / 100) * bar_width)

    if score > 70:
        color = (0, 255, 0)
    elif score > 40:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_height), color, -1)

    # update global variable for Flask
    global_attention_score = score

    cv2.imshow("Student Attention Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------
# 🎯 Final session summary
# ------------------------------------------------------
final_score = attention_sum / total_frames if total_frames > 0 else 0
print(f"\n🧠 Final Attention Score: {final_score:.2f}/100")

if final_score > 85:
    print("✅ Excellent focus!")
    speak("Excellent focus! Keep it up.")
elif final_score > 70:
    print("🙂 Good focus, minor distractions.")
    speak("Good focus, but stay sharp.")
elif final_score > 50:
    print("⚠️ Low focus, try to stay attentive.")
    speak("Hey! Pay attention. You're getting distracted.")
else:
    print("❌ Poor attention detected.")
    speak("You are not focused at all. Concentrate now!")

# helper for Flask
def get_attention_score():
    return global_attention_score
