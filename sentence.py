import cv2
import os

video_path = 'hello_video.mp4'  # Your recorded hello gesture video
output_dir = 'dataset/hello'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5  # extract one frame every 5 frames
count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:
        cv2.imwrite(os.path.join(output_dir, f"hello_{saved_count:03d}.jpg"), frame)
        saved_count += 1
    count += 1

cap.release()
print(f"Extracted {saved_count} frames.")
