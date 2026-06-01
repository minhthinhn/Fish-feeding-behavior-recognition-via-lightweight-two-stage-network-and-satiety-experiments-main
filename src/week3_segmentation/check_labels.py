import os

for split in ["train", "val", "test"]:

    label_dir = f"data/dataset/{split}/labels"

    print(f"\nChecking {split}...")

    for file in os.listdir(label_dir):

        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):

            parts = line.strip().split()

            # tối thiểu:
            # class x1 y1 x2 y2 x3 y3
            if len(parts) < 7:
                print(f"INVALID: {file} line {line_num+1}")
                break

            # số tọa độ phải chẵn
            if (len(parts) - 1) % 2 != 0:
                print(f"ODD_COORDS: {file} line {line_num+1}")
                break

print("\nDone!")