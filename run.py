# import os
# import string

# # Parent directory
# parent_dir = "asl_alphabet"

# # Create parent folder if not exists
# os.makedirs(parent_dir, exist_ok=True)

# # Loop A–Z and create each folder
# for letter in string.ascii_uppercase:
#     os.makedirs(os.path.join(parent_dir, letter), exist_ok=True)

# print("Folders A–Z created inside asl_alphabet/")


# import os

# parent_dir = "dataset"  # e.g., "asl_alphabet"

# # Loop over each folder inside the parent_dir
# for folder_name in os.listdir(parent_dir):
#     folder_path = os.path.join(parent_dir, folder_name)
    
#     if os.path.isdir(folder_path):
#         # Get list of image files sorted alphabetically
#         images = sorted(os.listdir(folder_path))
        
#         # If more than 300 images, delete images beyond 300
#         if len(images) > 300:
#             to_delete = images[300:]  # images from index 300 onward
#             for img_file in to_delete:
#                 img_path = os.path.join(folder_path, img_file)
#                 os.remove(img_path)
#             print(f"Deleted {len(to_delete)} images from {folder_name}")
#         else:
#             print(f"{folder_name} has {len(images)} images; no deletion needed")

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque

# Load model and labels (same as before)
MODEL_PATH = "models/asl_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

labels = [None] * len(class_indices)
for label, index in class_indices.items():
    labels[index] = label

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SMOOTHING_WINDOW = 5
pred_history = deque(maxlen=SMOOTHING_WINDOW)

sentence = ""      # Holds the built sentence
last_char = ""     # Last appended character (for debounce)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                hand_img = image[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                pred = model.predict(hand_img, verbose=0)
                pred_label = labels[np.argmax(pred)]
                pred_history.append(pred_label)

                most_common = max(set(pred_history), key=pred_history.count)

                # Debounce: add character only if different from last appended
                if most_common != last_char and most_common not in ['nothing', 'space', 'del']:
                    sentence += most_common
                    last_char = most_common

                # Handle space: add space if predicted 'space'
                if most_common == 'space' and (len(sentence) > 0 and sentence[-1] != ' '):
                    sentence += ' '

                # Handle delete: remove last character if predicted 'del'
                if most_common == 'del' and len(sentence) > 0:
                    sentence = sentence[:-1]
                    last_char = ''  # reset last_char so next letter can be appended

                # Draw landmarks and prediction
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(image, f'Predicted: {most_common}', (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the built sentence on screen
        cv2.putText(image, f'Sentence: {sentence}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('ASL Recognition', image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
