import cv2
import os
from pathlib import Path

# =========================
# Classes
# =========================
CLASSES = {
    "strong": 0,
    "middle": 1,
    "weak": 2,
    "none": 3
}

# =========================
# Paths
# =========================
DATASET_DIR = "dataset"
OUTPUT_DIR = "data/frames"

FRAME_INTERVAL = 30  # lấy 1 frame mỗi 30 frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Extract frames
# =========================
for class_name in CLASSES.keys():

    class_input = os.path.join(DATASET_DIR, class_name)
    class_output = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(class_output, exist_ok=True)

    videos = os.listdir(class_input)

    for video_name in videos:

        video_path = os.path.join(class_input, video_name)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        saved_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_INTERVAL == 0:

                save_name = f"{video_name[:-4]}_{saved_id:06d}.jpg"

                save_path = os.path.join(class_output, save_name)

                cv2.imwrite(save_path, frame)

                saved_id += 1

            frame_id += 1

        cap.release()

        print(f"Finished: {video_name}")

print("ALL FRAMES EXTRACTED")