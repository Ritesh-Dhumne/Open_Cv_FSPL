import cv2
import numpy as np

def detect_color(frame, color='red'):
    """Detects and isolates the specified color while converting other colors to grayscale."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue
    if color == 'red':
        lower1, upper1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower2, upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    elif color == 'blue':
        lower1, upper1 = np.array([100, 150, 0]), np.array([140, 255, 255])
        lower2, upper2 = np.array([100, 150, 0]), np.array([140, 255, 255])
    else:
        return frame  # Return original frame if an invalid color is selected

    # Create masks to detect selected color
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2

    # Isolate the selected color
    color_isolated = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the rest of the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    background = cv2.bitwise_and(gray_colored, gray_colored, mask=cv2.bitwise_not(mask))

    # Combine color-isolated and grayscale parts
    result = cv2.add(color_isolated, background)
    return result

# Open webcam
cap = cv2.VideoCapture(0)
color_mode = 'red'  # Default detection color

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame based on the selected color
    processed_frame = detect_color(frame, color_mode)

    # Display the result
    cv2.imshow('Color Detection', processed_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Exit on 'q'
    elif key == ord('r'):
        color_mode = 'red'  # Switch to red detection
    elif key == ord('b'):
        color_mode = 'blue'  # Switch to blue detection

# Release resources
cap.release()
cv2.destroyAllWindows()




