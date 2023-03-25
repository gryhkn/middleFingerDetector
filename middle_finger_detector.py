import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def change_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = (hsv[:, :, 2] * 0.7).astype(np.uint8)
    new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_image

def draw_busted_label(image):
    height, width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Busted"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height - text_size[1]) // 2
    cv2.rectangle(image, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + text_size[1] + 20), (0, 0, 255), 2)
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(1)
_, first_frame = cap.read()
height, width, _ = first_frame.shape

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Kameradan görüntü alınamıyor. Lütfen kamerayı kontrol edin.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                if (middle_finger_tip.y < middle_finger_pip.y and
                        index_finger_tip.y > middle_finger_pip.y and
                        ring_finger_tip.y > middle_finger_pip.y and
                        pinky_tip.y > middle_finger_pip.y):
                    frame = change_colors(frame)
                    draw_busted_label(frame)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hands', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
       
