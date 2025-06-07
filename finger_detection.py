import cv2
import numpy as np

def count_fingers(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (35, 35), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the hand)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        # Create a convex hull around the largest contour
        hull = cv2.convexHull(max_contour)

        # Find convexity defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_indices)

        # Count fingers based on convexity defects
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate the angle between fingers
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                # Count fingers if the angle is less than 90 degrees
                if angle <= np.pi / 2:
                    finger_count += 1

        return finger_count + 1  # Add 1 for the thumb
    return 0

# Start video capture from the default camera (0)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Count fingers
    finger_count = count_fingers(frame)

    # Display the number of fingers on the frame
    cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output in the console
    print(f"Fingers detected: {finger_count}")

    # Display the resulting frame
    cv2.imshow('Finger Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
