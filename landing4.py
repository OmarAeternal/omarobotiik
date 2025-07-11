import cv2
import numpy as np

def detect_red_box_and_area(frame, expand_ratio=3.0):
    """
    Detect the largest red contour, return center of an expanded bounding box and coverage.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color ranges
    lower1, upper1 = np.array([0,120,70]), np.array([10,255,255])
    lower2, upper2 = np.array([170,120,70]), np.array([180,255,255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Compute coverage
    red_area = cv2.countNonZero(mask)
    frame_area = frame.shape[0] * frame.shape[1]
    coverage = red_area / frame_area

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, coverage

    # Choose largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:
        return None, None, coverage

    # Get bounding box and expand it
    x, y, w, h = cv2.boundingRect(c)
    # Expand around center
    cx, cy = x + w//2, y + h//2
    new_w, new_h = int(w * expand_ratio), int(h * expand_ratio)
    # Compute expanded top-left
    x_exp = max(0, cx - new_w//2)
    y_exp = max(0, cy - new_h//2)
    # Compute expanded center (unchanged)
    return cx, cy, coverage, (x_exp, y_exp, new_w, new_h)

def compute_command_and_descent(cx, cy, coverage, frame_shape, deadzone=20):
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w//2, h//2
    if cx is None or cy is None:
        return "ascending", None
    dx, dy = cx - cx_frame, cy - cy_frame
    if abs(dy) > deadzone:
        return ("move forward" if dy < 0 else "move backward"), None
    if abs(dx) > deadzone:
        return ("move left" if dx > 0 else "move right"), None
    stage = min(int(coverage * 20) + 1, 20)
    if stage >= 20:
        return "landed", 20
    return "descending", stage

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, coverage, bbox = detect_red_box_and_area(frame)
        cmd, stage = compute_command_and_descent(cx, cy, coverage, frame.shape)

        # Draw guides
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2,0),(w//2,h),(255,255,0),1)
        cv2.line(frame, (0,h//2),(w,h//2),(255,255,0),1)

        # Draw expanded bounding box
        if bbox:
            x_exp, y_exp, new_w, new_h = bbox
            cv2.rectangle(frame, (x_exp, y_exp), (x_exp+new_w, y_exp+new_h), (255,0,255), 2)

        # Draw center
        if cx is not None and cy is not None:
            cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)

        # Choose notification color
        if cmd.startswith("move") or cmd == "ascending":
            color = (0,0,255)
        elif cmd == "descending":
            color = (255,0,0)
        else:
            color = (0,255,0)

        text = cmd if stage is None else f"{cmd} stage {stage}/20"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        cv2.imshow("Precision Landing", frame)
        if cmd == "landed":
            print("Landed successfully.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
