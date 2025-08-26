import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("models/asl_model.h5")
labels = list("ABCDdelEFGHIKLMNnothingOPQRSspaceTUVWXYZ")  # Adjust to match your training

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to normalize hand based on landmarks
def normalize_hand(image, landmarks, img_size=64):
    # Get all landmark coordinates
    h, w, _ = image.shape
    coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    # Get bounding box around landmarks
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    # Add margin
    margin = 0.2 * max(x_max - x_min, y_max - y_min)
    x_min = max(int(x_min - margin), 0)
    y_min = max(int(y_min - margin), 0)
    x_max = min(int(x_max + margin), w)
    y_max = min(int(y_max + margin), h)

    # Crop
    hand_crop = image[y_min:y_max, x_min:x_max]

    if hand_crop.size == 0:
        return None

    # Resize to model input
    hand_resized = cv2.resize(hand_crop, (img_size, img_size))
    hand_resized = hand_resized.astype("float32") / 255.0
    return np.expand_dims(hand_resized, axis=0)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Normalize using landmarks
                hand_img = normalize_hand(frame, hand_landmarks.landmark)

                if hand_img is not None:
                    preds = model.predict(hand_img, verbose=0)[0]
                    pred_idx = np.argmax(preds)
                    confidence = preds[pred_idx]

                    if confidence > 0.6:  # only trust if confident
                        pred_label = labels[pred_idx]
                        h, w, _ = frame.shape
                        x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                        y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                        cv2.putText(frame, f"{pred_label} ({confidence:.2f})", 
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
