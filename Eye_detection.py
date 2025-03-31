import cv2

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (Haar cascades work better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Define face region
        face_roi = gray[y:y + h, x:x + w]
        frame_roi = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around detected eyes
            cv2.rectangle(frame_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show video feed with detected faces and eyes
    cv2.imshow('Eye Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
