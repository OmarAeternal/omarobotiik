import cv2
import numpy as np

def detect_red_box_and_area(frame):
    """
    Detect the largest red box in the frame, return its center and red area fraction.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color ranges
    lower1, upper1 = np.array([0,120,70]), np.array([10,255,255])
    lower2, upper2 = np.array([170,120,70]), np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Morphology clean
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Compute overall red coverage
    red_area = cv2.countNonZero(mask)
    frame_area = frame.shape[0] * frame.shape[1]
    coverage = red_area / frame_area

    # Find largest contour for position
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if cv2.contourArea(c) >= 500:
            return x+w//2, y+h//2, coverage
    return None, None, coverage

def compute_command_and_descent(cx, cy, coverage, frame_shape, deadzone=20):
    h, w = frame_shape[:2]
    cx_frame, cy_frame = w//2, h//2

    # Horizontal/vertical adjustment
    if cx is None or cy is None:
        return "searching", None
    dx, dy = cx - cx_frame, cy - cy_frame
    if abs(dy) > deadzone:
        return ("move forward" if dy < 0 else "move backward"), None
    if abs(dx) > deadzone:
        return ("move left" if dx > 0 else "move right"), None

    # Centered: handle descent stages
    stages = int(coverage * 6)  # 0 to 6
    if stages >= 6:
        return "landed", 6
    elif stages > 0:
        return "descending", stages
    else:
        return "centered", 0

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx, cy, coverage = detect_red_box_and_area(frame)
        cmd, stage = compute_command_and_descent(cx, cy, coverage, frame.shape)

        # Draw visuals
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2,0),(w//2,h),(255,255,0),1)
        cv2.line(frame, (0,h//2),(w,h//2),(255,255,0),1)
        if cx and cy:
            cv2.circle(frame,(cx,cy),8,(0,255,0),-1)

        # Display command and stage
        text = cmd
        if cmd == "descending":
            text += f" stage {stage}/6"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

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
