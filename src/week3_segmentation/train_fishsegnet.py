from ultralytics import YOLO


def main():

    # Load custom architecture
    model = YOLO("models/fishsegnet_prl.yaml")

    # Train
    model.train(
        data="D:/Fish-feeding-behavior-recognition-via-lightweight-two-stage-network-and-satiety-experiments-main/data/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,   # QUAN TRỌNG
        project="runs/fishsegnet",
        name="fishsegnet_prl_v1"
    )


if __name__ == "__main__":
    main()