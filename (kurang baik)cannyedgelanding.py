import cv2
import numpy as np

class KalmanTracker:
    def __init__(self):
        # State: [x, y, dx, dy]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        cv2.setIdentity(self.kalman.processNoiseCov, 1e-4)
        cv2.setIdentity(self.kalman.measurementNoiseCov, 1e-1)

    def update(self, cx, cy):
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(meas)
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])


def detect_refined_red(frame, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2):
    # Convert to HSV and threshold for red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, hsv_lower1, hsv_upper1),
        cv2.inRange(hsv, hsv_lower2, hsv_upper2)
    )

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Combine mask and edges
    refined = cv2.bitwise_and(mask, edges)
    refined = cv2.dilate(refined, np.ones((5,5), np.uint8), iterations=1)

    # Find contours
    cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            # compute coverage on raw mask
            area = cv2.countNonZero(mask)
            coverage = area / (frame.shape[0] * frame.shape[1])
            return cx, cy, coverage, refined
    # No detection
    area = cv2.countNonZero(mask)
    coverage = area / (frame.shape[0] * frame.shape[1])
    return None, None, coverage, refined


def compute_command(cx, cy, coverage, frame_shape, deadzone=20):
    h, w = frame_shape[:2]
    center_x, center_y = w//2, h//2
    if cx is None:
        return "Searching", None
    dx, dy = cx - center_x, cy - center_y
    if abs(dy) > deadzone:
        return ("move forward" if dy < 0 else "move backward"), None
    if abs(dx) > deadzone:
        return ("move left" if dx > 0 else "move right"), None
    # descent logic
    stage = min(int(coverage * 20) + 1, 20)
    if stage >= 20:
        return "landed", 20
    return "descending", stage


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # HSV ranges for red
    lower1, upper1 = np.array([0,120,70]), np.array([10,255,255])
    lower2, upper2 = np.array([170,120,70]), np.array([180,255,255])

    tracker = KalmanTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, coverage, mask = detect_refined_red(
            frame, lower1, upper1, lower2, upper2
        )
        # Use Kalman to smooth
        if cx is not None:
            cx, cy = tracker.update(cx, cy)

        cmd, stage = compute_command(cx, cy, coverage, frame.shape)

        # Draw overlays
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2,0), (w//2,h), (255,255,0), 1)
        cv2.line(frame, (0,h//2), (w,h//2), (255,255,0), 1)
        if cx is not None:
            cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)
        # Display mask for debugging
        cv2.imshow("Mask Refined", mask)

        # Text color
        if cmd.startswith("move") or cmd == "Searching":
            color = (0,0,255)
        elif cmd == "descending":
            color = (255,0,0)
        else:
            color = (0,255,0)
        text = cmd if stage is None else f"{cmd} stage {stage}/20"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Precision Landing v2", frame)
        key = cv2.waitKey(1) & 0xFF
        if cmd == "landed":
            print("Landed successfully.")
            break
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
