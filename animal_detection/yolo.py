from ultralytics import YOLO


def train_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26s.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-s_oai_2026-02-07")


def train_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26m.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-m_oai_2026-02-07")


def resume_training() -> None:
    model = YOLO("/mnt/data/afarec/code/runs/detect/rtdetrv1-l_oai_2026-02-02/weights/last.pt")
    results = model.train(resume=True)
    # print(results)


if __name__ == '__main__':
    train_yolo26s()
    # resume_training()