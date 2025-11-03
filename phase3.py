import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# --- Function to calculate Mouth Aspect Ratio (MAR) ---
def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[50 - 48], mouth_points[58 - 48])  # p51 - p59
    B = distance.euclidean(mouth_points[52 - 48], mouth_points[56 - 48])  # p53 - p57
    C = distance.euclidean(mouth_points[49 - 48], mouth_points[55 - 48])  # p50 - p58
    D = distance.euclidean(mouth_points[48 - 48], mouth_points[54 - 48])  # p49 - p55 (horizontal)
    mar = (A + B + C) / (3.0 * D)
    return mar

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# Thresholds and counters
MAR_THRESHOLD = 0.60      # Above this = mouth open (possible yawn)
CONSEC_FRAMES_YAWN = 15   # Must stay open for these many frames
yawn_counter = 0
total_yawns = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        # Get mouth region
        mouth = points[48:60]

        # Calculate MAR
        mar = mouth_aspect_ratio(mouth)

        # Draw yellow mouth outline
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)

        # Yawn detection logic
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            if yawn_counter >= CONSEC_FRAMES_YAWN:
                total_yawns += 1
            yawn_counter = 0

        # Display MAR & total yawns
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawn Count: {total_yawns}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Phase 3 - Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
