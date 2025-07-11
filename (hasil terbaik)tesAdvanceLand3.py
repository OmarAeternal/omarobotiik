import cv2
import numpy as np

def detect_red_box_and_area(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0,120,70]), np.array([10,255,255])
    lower2, upper2 = np.array([170,120,70]), np.array([180,255,255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
    # Morphology: close, open
    kernel_small = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    kernel_large = np.ones((55,55), np.uint8)
    mask = cv2.dilate(mask, kernel_large, iterations=1)

    red_area = cv2.countNonZero(mask)
    frame_area = frame.shape[0] * frame.shape[1]
    coverage = red_area / frame_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) >= 500:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            return cx, cy, coverage
    return None, None, coverage
def compute_command_and_descent(cx, cy, coverage, frame_shape, deadzone=20):
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w//2, h//2
    if cx is None or cy is None:
        return "Searching", None
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
        cx, cy, coverage = detect_red_box_and_area(frame)
        cmd, stage = compute_command_and_descent(cx, cy, coverage, frame.shape)
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2,0),(w//2,h),(255,255,0),1)
        cv2.line(frame, (0,h//2),(w,h//2),(255,255,0),1)
        if cx is not None and cy is not None:
            cv2.circle(frame,(cx,cy),12,(0,255,0),-1)
        if cmd.startswith("move") or cmd == "Searching":
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
