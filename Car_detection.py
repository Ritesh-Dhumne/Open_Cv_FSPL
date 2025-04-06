import cv2

# Load the pre-trained Haar cascade classifier
car_cascade = cv2.CascadeClassifier(r'D:\python\Open_Cv_FSPL\Harcasscade\haarcascade_car.xml')

# Use a video file or 0 for webcam
video_path = 0  # Replace with 0 to use the webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Focus on bottom half of the frame
    roi_gray = gray[gray.shape[0]//2:, :]

    # Detect cars with improved parameters
    cars = car_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60),
        maxSize=(300, 300)
    )

    # Draw rectangles on original frame (adjust y for ROI offset)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y + gray.shape[0]//2), (x + w, y + h + gray.shape[0]//2), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Vehicle Detection (Improved)', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
