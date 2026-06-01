import os

DATASET_DIR = "data/dataset"
SPLITS = ["train", "val", "test"]

total_removed = 0
total_files_modified = 0

for split in SPLITS:

    label_dir = os.path.join(
        DATASET_DIR,
        split,
        "labels"
    )

    print(f"\nChecking {split}...")

    for file in os.listdir(label_dir):

        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        valid_lines = []
        removed = 0

        for idx, line in enumerate(lines, start=1):

            parts = line.strip().split()

            # Segmentation phải có:
            # class + ít nhất 3 điểm
            if len(parts) >= 7:
                valid_lines.append(line)
            else:
                print(
                    f"INVALID: {file} line {idx}"
                )

                removed += 1
                total_removed += 1

        if removed > 0:

            with open(path, "w") as f:
                f.writelines(valid_lines)

            total_files_modified += 1

            print(
                f"FIXED: {file} "
                f"(removed {removed} objects)"
            )

print("\n===================")
print("DONE")
print("Files modified:", total_files_modified)
print("Objects removed:", total_removed)
print("===================")