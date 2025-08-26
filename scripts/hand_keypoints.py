import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- Settings ---
SAVE_SAMPLES = True          # Set True if you want to save keypoints when pressing 's'
SAMPLES_DIR = "../dataset/custom_samples"  # where samples will be saved (relative to scripts/)
MAX_SAVED = 1000              # safety cap

os.makedirs(SAMPLES_DIR, exist_ok=True)

# MediaPipe utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Buffer to hold recent predictions / keypoints for simple smoothing later
KEYPOINT_BUFFER = []
BUFFER_SIZE = 8

def landmark_list_to_array(landmark_list, image_width, image_height):
    """
    Convert mediapipe landmark list to an array of shape (21, 3)
    where columns are [x_norm, y_norm, z_norm] in normalized coordinates,
    and returns pixel coords too if needed.
    """
    arr_norm = np.zeros((21, 3), dtype=np.float32)
    arr_px = np.zeros((21, 2), dtype=np.int32)
    for i, lm in enumerate(landmark_list.landmark):
        arr_norm[i, 0] = lm.x
        arr_norm[i, 1] = lm.y
        arr_norm[i, 2] = lm.z  # z is relative depth
        arr_px[i, 0] = int(lm.x * image_width)
        arr_px[i, 1] = int(lm.y * image_height)
    return arr_norm, arr_px

def save_keypoints(np_array_norm, label="unknown"):
    """Save one sample to disk as compressed npz with label and timestamp."""
    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.npz"
    path = os.path.join(SAMPLES_DIR, filename)
    # data saved: normalized keypoints and label
    np.savez_compressed(path, keypoints=np_array_norm.astype(np.float32), label=label)
    print(f"[SAVED] {path}")

def main():
    global SAVE_SAMPLES
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions and index.")
        return

    # MediaPipe Hands: realtime mode, 1 hand, good tradeoffs for detection/tracking
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    print("Running hand detection. Press 'q' to quit, 's' to save keypoints (if SAVE_SAMPLES=True).")

    saved_count = 0
    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Using the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks and connections on original BGR frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

            # Convert landmarks to arrays
            arr_norm, arr_px = landmark_list_to_array(hand_landmarks, w, h)

            # Print (or log) — here we print the first few points for demo
            # Uncomment if you want the full array printed every frame (can be verbose)
            # print("Normalized keypoints (21x3):\n", arr_norm)
            # print("Pixel coords (21x2):\n", arr_px)

            # Add to smoothing buffer
            KEYPOINT_BUFFER.append(arr_norm)
            if len(KEYPOINT_BUFFER) > BUFFER_SIZE:
                KEYPOINT_BUFFER.pop(0)

            # Example: simple smoothing by averaging buffer
            smoothed = np.mean(KEYPOINT_BUFFER, axis=0)

            # Display the index+pixel coords of a few landmark points (thumb tip, index tip, middle tip)
            thumb_tip = arr_px[4]
            index_tip = arr_px[8]
            middle_tip = arr_px[12]
            cv2.circle(frame, tuple(thumb_tip), 6, (255, 0, 255), -1)
            cv2.circle(frame, tuple(index_tip), 6, (255, 0, 255), -1)
            cv2.circle(frame, tuple(middle_tip), 6, (255, 0, 255), -1)

            # Show confidence-like info (z coordinate is relative depth)
            # lower z generally means closer to camera in MediaPipe's coordinate system
            cv2.putText(frame, f"z-thumb:{arr_norm[4,2]:.2f}", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"z-index:{arr_norm[8,2]:.2f}", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # OPTIONAL: when SAVE_SAMPLES True, press 's' to save current normalized keypoints with a label
            # (you will be prompted to enter label in console)
        else:
            # No hand found: clear or keep buffer depending on design
            # KEYPOINT_BUFFER.clear()
            pass

        # FPS display
        now = time.time()
        fps = 1.0 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("ASL - Hand Keypoints (press q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and SAVE_SAMPLES:
            if results and results.multi_hand_landmarks:
                label = input("Enter label for this sample (e.g., A, B, hello): ").strip()
                if label == "":
                    label = "unknown"
                save_keypoints(arr_norm, label=label)
                saved_count += 1
                if saved_count >= MAX_SAVED:
                    print("Saved cap reached MAX_SAVED. Stopping saves.")
                    SAVE_SAMPLES = False
            else:
                print("No hand detected — can't save sample.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
