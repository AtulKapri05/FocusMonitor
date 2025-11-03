import cv2
import dlib
import numpy as np

# Load face detector and 68 landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3D model points of key facial landmarks (nose, eyes, mouth corners, chin)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        # 2D image points from landmarks (correspond to model_points)
        image_points = np.array([
            (points[30][0], points[30][1]),     # Nose tip
            (points[8][0], points[8][1]),       # Chin
            (points[36][0], points[36][1]),     # Left eye left corner
            (points[45][0], points[45][1]),     # Right eye right corner
            (points[48][0], points[48][1]),     # Left Mouth corner
            (points[54][0], points[54][1])      # Right Mouth corner
        ], dtype=np.float64)

        # Camera parameters
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # SolvePnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Project a line from the nose tip (to visualize direction)
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )

        # Draw the face direction line
        p1 = (int(image_points[0][0]), int(image_points[0][1]))  # Nose tip
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))  # End point
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # Display text based on angle direction (optional)
        cv2.putText(frame, "Head Pose Estimation", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Phase 4 - Head Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
