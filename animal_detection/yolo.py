from ultralytics import YOLO


def train_model() -> None:
    model = YOLO("yolo11s.pt")

    results = model.train(data="/mnt/data/afarec/data/OpenAnimalImages/dataset.yaml",
                          epochs=100, imgsz=640, device="cuda",
                          save=True, save_period=1,
                          name="yolo11s_oai_2025-12-27")
    print(results)
    pass


if __name__ == '__main__':
    train_model()