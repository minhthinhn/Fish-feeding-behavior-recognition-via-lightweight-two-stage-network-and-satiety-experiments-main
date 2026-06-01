import os
import random
import shutil
from pathlib import Path

random.seed(42)

DATASET_DIR = "data/dataset"

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

random.shuffle(images)

n = len(images)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

for split in splits:

    os.makedirs(
        os.path.join(DATASET_DIR, split, "images"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(DATASET_DIR, split, "labels"),
        exist_ok=True
    )

    for img_file in splits[split]:

        label_file = Path(img_file).stem + ".txt"

        shutil.copy(
            os.path.join(IMAGE_DIR, img_file),
            os.path.join(DATASET_DIR, split, "images", img_file)
        )

        src_label = os.path.join(LABEL_DIR, label_file)

        if os.path.exists(src_label):
            shutil.copy(
                src_label,
                os.path.join(DATASET_DIR, split, "labels", label_file)
            )

print("Dataset split completed.")