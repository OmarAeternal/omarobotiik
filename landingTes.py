import cv2
import numpy as np


def detect_red_box(frame):
    """
    Detect the largest red box in the frame and return its center coordinates.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range (two ranges to cover wrap-around)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Threshold
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Choose the largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:  # minimum area threshold
        return None, None

    # Compute center of bounding box
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def compute_command(cx, cy, frame_shape, deadzone=20):
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2

    if cx is None or cy is None:
        return "searching"

    dx = cx - center_x
    dy = cy - center_y

    if abs(dy) > deadzone:
        return "move forward" if dy < 0 else "move backward"
    if abs(dx) > deadzone:
        return "move left" if dx > 0 else "move right"
    return "centered"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy = detect_red_box(frame)
        cmd = compute_command(cx, cy, frame.shape)

        # Draw visualization
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
        if cx and cy:
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(frame, cmd, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Red Box Precision Landing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
