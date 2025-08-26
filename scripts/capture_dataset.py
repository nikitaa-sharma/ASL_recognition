import cv2
import mediapipe as mp
import os
import string

# --- Settings ---
OUTPUT_DIR = "dataset"
IMG_SIZE = 64  # match your model's input
CAPTURE_DELAY = 2  # capture every N frames

# Class list (A–Z + extras)
CLASSES = list(string.ascii_uppercase) + ["nothing", "space", "del"]

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Create dataset folder and subfolders
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

cap = cv2.VideoCapture(0)
frame_count = 0
current_label = None

print("\n=== Dataset Capture Controls ===")
print("Press A–Z for letters")
print("Press 1 for 'nothing'")
print("Press 2 for 'space'")
print("Press 3 for 'del'")
print("Press 4 for 'Hello'")
print("Press ESC to quit")
print("==============================\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

                # Save image if label is selected
                if current_label and frame_count % CAPTURE_DELAY == 0:
                    save_path = os.path.join(OUTPUT_DIR, current_label,
                                             f"{current_label}_{len(os.listdir(os.path.join(OUTPUT_DIR, current_label)))}.jpg")
                    cv2.imwrite(save_path, hand_img)
                    cv2.putText(frame, f"Saved: {current_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        # For "nothing" class, capture without a hand
        if current_label == "nothing" and frame_count % CAPTURE_DELAY == 0:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            save_path = os.path.join(OUTPUT_DIR, current_label,
                                     f"{current_label}_{len(os.listdir(os.path.join(OUTPUT_DIR, current_label)))}.jpg")
            cv2.imwrite(save_path, resized)
            cv2.putText(frame, f"Saved: nothing", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Dataset Capture", frame)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z
        current_label = chr(key).upper()
        print(f"Capturing for letter: {current_label}")
    elif key == ord('1'):
        current_label = "nothing"
        print("Capturing for: nothing")
    elif key == ord('2'):
        current_label = "space"
        print("Capturing for: space")
    elif key == ord('3'):
        current_label = "del"
        print("Capturing for: del")
    elif key == ord('4'):
        current_label = "Hello"
        print("Capturing for: Hello")

cap.release()
cv2.destroyAllWindows()
