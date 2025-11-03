import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# ---- Function to calculate Eye Aspect Ratio (EAR) ----
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])  # p2–p6 vertical
    B = distance.euclidean(eye_points[2], eye_points[4])  # p3–p5 vertical
    C = distance.euclidean(eye_points[0], eye_points[3])  # p1–p4 horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# ---- Load models ----
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# ---- Blink logic parameters ----
EAR_THRESHOLD = 0.22      # 0.21–0.25 works best depending on your face distance
CONSEC_FRAMES = 2         # blink registered after these many closed frames
blink_counter = 0
total_blinks = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Draw a green rectangle around the face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        # Get both eyes
        left_eye = points[36:42]
        right_eye = points[42:48]

        # Draw blue eye contours
        cv2.polylines(frame, [left_eye], True, (255, 0, 0), 1)
        cv2.polylines(frame, [right_eye], True, (255, 0, 0), 1)

        # Compute EAR
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        avgEAR = (leftEAR + rightEAR) / 2.0

        # Blink detection logic
        if avgEAR < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

        # Display EAR & Blink Count
        cv2.putText(frame, f"EAR: {avgEAR:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Phase 2 - Stable Blink Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
