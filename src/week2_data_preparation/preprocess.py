import cv2
import os
from tqdm import tqdm

# =========================
# Classes
# =========================
CLASSES = ["strong", "middle", "weak", "none"]

# =========================
# Paths
# =========================
INPUT_DIR = "data/frames"
OUTPUT_DIR = "data/preprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CLAHE
# =========================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# =========================
# Process Images
# =========================
for class_name in CLASSES:

    input_class_dir = os.path.join(INPUT_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(output_class_dir, exist_ok=True)

    image_files = os.listdir(input_class_dir)

    for image_name in tqdm(image_files, desc=class_name):

        image_path = os.path.join(input_class_dir, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # =========================
        # Convert to LAB
        # =========================
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        # =========================
        # Apply CLAHE
        # =========================
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # =========================
        # Denoising
        # =========================
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            10,
            10,
            7,
            21
        )

        # =========================
        # Save
        # =========================
        save_path = os.path.join(output_class_dir, image_name)

        cv2.imwrite(save_path, denoised)

print("PREPROCESSING COMPLETED")