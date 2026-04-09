import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

   
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # improve performance
    rgb_frame.flags.writeable = False
    result = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            finger_tips = [4, 8, 12, 16, 20]
            fingers = []

            # Thumb (special case)
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for tip in finger_tips[1:]:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            print("Fingers:", fingers)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8: # for index finger
                    print("Index finger:", cx, cy)

                cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()