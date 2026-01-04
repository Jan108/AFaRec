from ultralytics import YOLO


def train_model() -> None:
    model = YOLO("yolo12s.pt")

    results = model.train(data="/mnt/data/afarec/data/OpenAnimalImages/dataset.yaml",
                          epochs=100, imgsz=640, device="cuda",
                          save=True, save_period=1,
                          name="yolo12s_oai_2025-12-29")
    # print(results)
    pass


def resume_training() -> None:
    model = YOLO("/mnt/data/afarec/code/runs/detect/yolo12s_oai_2025-12-29/weights/last.pt")
    results = model.train(resume=True)
    # print(results)


if __name__ == '__main__':
    # train_model()
    resume_training()