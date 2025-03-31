import cv2

# Load Haar cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Define face region
        face_roi = gray[y:y + h, x:x + w]
        frame_roi = frame[y:y + h, x:x + w]

        # Detect smiles within the face region
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))

        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around the detected smile
            cv2.rectangle(frame_roi, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

            # Display "Smiling" text when a smile is detected
            cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video feed with detected faces and smiles
    cv2.imshow('Smile Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
