from ultralytics import YOLO

def train_model() -> None:
    model = YOLO("yolo11s.pt")

    results = model.train(data="/mnt/data/afarec/data/OpenAnimalImages/dataset.yaml", epochs=100, imgsz=640)
    print(results)
    pass


if __name__ == '__main__':
    train_model()