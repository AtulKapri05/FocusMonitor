import cv2
import dlib

# Load predefined models
detector = dlib.get_frontal_face_detector()   # Detects faces in the image
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Predicts 68 facial landmarks

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:                      # Infinite loop for continuous capture
    ret, frame = cap.read()      # Reads one frame from the webcam
    if not ret:                  # If frame not captured correctly, stop the loop
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Convert colored frame to grayscale
    faces = detector(gray)        # Detect all faces in the grayscale image

    for face in faces:            # Loop through each detected face
        x1, y1 = face.left(), face.top()             # Top-left corner of the face box
        x2, y2 = face.right(), face.bottom()         # Bottom-right corner of the face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle around the face

        landmarks = predictor(gray, face)            # Predict 68 facial landmarks on the detected face

        for n in range(0, 68):                       # Loop through all 68 landmarks
            x = landmarks.part(n).x                  # X-coordinate of nth landmark
            y = landmarks.part(n).y                  # Y-coordinate of nth landmark
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Draw a small red dot at that landmark

    cv2.imshow("Phase 1 – Face and Landmarks", frame)     # Display live webcam output

    if cv2.waitKey(1) & 0xFF == ord('q'):            # If 'q' pressed → exit loop
        break

# When loop ends
cap.release()               # Release the webcam
cv2.destroyAllWindows()     # Close all OpenCV windows
