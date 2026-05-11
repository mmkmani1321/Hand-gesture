import cv2
import numpy as np
import mediapipe as mp
import time

# ------------------ Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None

prev_x, prev_y = 0, 0

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

last_color_change = 0

# ------------------ Finger Count ------------------
def count_fingers(hand_landmarks):
    fingers = []

    # Thumb (simple logic)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

# ------------------ Fist Detection ------------------
def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            return False
    return True

# ------------------ Main Loop ------------------
while True:
    success, frame = cap.read()
    if not success:
        continue

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            finger_count = count_fingers(hand_landmarks)

            # ✊ ERASE (priority)
            if is_fist(hand_landmarks):
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                cv2.putText(frame, "ERASE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prev_x, prev_y = 0, 0

            # ✌️ COLOR CHANGE
            elif finger_count == 2:
                current_time = time.time()
                if current_time - last_color_change > 1:
                    color_index = (color_index + 1) % len(colors)
                    last_color_change = current_time

                cv2.putText(frame, "COLOR", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_index], 2)
                prev_x, prev_y = 0, 0

            # ☝️ DRAW
            elif finger_count == 1:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         colors[color_index], 5)

                prev_x, prev_y = x, y

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas + frame
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Paint", frame)

    key = cv2.waitKey(1)

    # Clear
    if key == ord('c'):
        canvas = np.zeros_like(frame)

    # Save
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()