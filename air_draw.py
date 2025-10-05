#source airpaint_env/Scripts/activate 
import cv2
import mediapipe as mp
import numpy as np

# ------------------- Initialization -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None

# Colors (BGR format)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]  # Red, Green, Blue, Yellow, White (eraser)
color_index = 0
brush_thickness = 5

# ------------------- Helper Function -------------------
def fingers_up(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_fold_status = []
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_fold_status.append(1)
        else:
            finger_fold_status.append(0)
    return finger_fold_status

prev_x, prev_y = 0, 0

# ------------------- Main Loop -------------------
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Mirror
    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, c = img.shape
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)  # Index fingertip

            fingers = fingers_up(hand_landmarks)

            # Gesture: ✋ (all fingers up) → clear canvas
            if all(fingers):
                canvas = np.zeros_like(img)

            # Gesture: ✌️ (index + middle up, others down) → switch color
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                color_index = (color_index + 1) % len(colors)

            # Drawing mode (only index finger up)
            elif fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), colors[color_index], brush_thickness)
                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ------------------- Overlay Canvas -------------------
    img = cv2.add(canvas, img)
  # Merge canvas directly

    # Display current color
    cv2.rectangle(img, (0, 0), (100, 100), colors[color_index], -1)

    cv2.imshow("Air Drawing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
