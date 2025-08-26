import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque

# Load trained model
MODEL_PATH = "models/asl_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices to map prediction index → label
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Build labels list: index to label mapping
labels = [None] * len(class_indices)
for label, index in class_indices.items():
    labels[index] = label

print("Loaded labels:", labels)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# For smoothing predictions over several frames
SMOOTHING_WINDOW = 5
pred_history = deque(maxlen=SMOOTHING_WINDOW)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)

        image_rgb.flags.writeable = True
        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

                # Clamp bounding box to frame dimensions
                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                hand_img = image[y_min:y_max, x_min:x_max]

                if hand_img.size == 0:
                    continue  # Skip invalid regions

                # Convert BGR to RGB, resize, normalize
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

                # Predict
                pred = model.predict(hand_img, verbose=0)
                pred_label = labels[np.argmax(pred)]
                pred_history.append(pred_label)

                # Smooth prediction (majority vote)
                most_common = max(set(pred_history), key=pred_history.count)

                # Draw hand landmarks & predicted label on frame
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, most_common, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ASL Recognition', image)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
