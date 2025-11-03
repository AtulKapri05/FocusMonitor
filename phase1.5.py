import cv2                       # OpenCV for camera and image processing
import dlib                      # Dlib for face and landmark detection
import time                      # Used to calculate FPS
import numpy as np               # Used for handling landmark points as arrays

# Load models
detector = dlib.get_frontal_face_detector()     # Pre-trained face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68 points model

cap = cv2.VideoCapture(0)  # open default webcam
prev_time = 0              # to store previous frame time

while True:
    ret, frame = cap.read()               # capture frame by frame
    if not ret:                           # if camera fails to capture, break loop
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Convert colored frame to grayscale
    faces = detector(gray)        # Detect all faces in the grayscale image

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()  # get coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle

        landmarks = predictor(gray, face)  # get all expressions
        points = []  # stores all points (empty list)

        for n in range(0, 68):  # looping through 68 points
            x = landmarks.part(n).x   # Get x coordinates ✅ (part, not parts)
            y = landmarks.part(n).y   # Get y coordinates ✅ (same)
            points.append((x, y))     # append coordinates to points
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # draw small red dots

        points = np.array(points)  # Convert to array ✅ (this line should be OUTSIDE the loop)

        # Draw outlines
        cv2.polylines(frame, [points[36:42]], True, (255, 0, 0), 1)   # Left Eye Blue
        cv2.polylines(frame, [points[42:48]], True, (255, 0, 0), 1)   # Right Eye Blue
        cv2.polylines(frame, [points[48:60]], True, (0, 255, 255), 1) # Mouth Yellow

    # --- FPS Calculation ---
    curr_time = time.time()  # Current time
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0  # FPS formula
    prev_time = curr_time    # Update previous frame time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Show FPS on screen (top-left corner)

    cv2.imshow("Phase 1.5 - Enhanced Tracking", frame)  # Show webcam with face landmarks + outlines + FPS

    if cv2.waitKey(1) & 0xFF == ord('q'):  # if q is pressed quit
        break

cap.release()                # Release Webcam
cv2.destroyAllWindows()      # Close all OpenCV windows ✅ (spelling fix)
