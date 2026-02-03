from ultralytics import YOLO
from ultralytics.models.rtdetr import RTDETR


def train_model() -> None:
    model = RTDETR("/mnt/data/afarec/code/runs/detect/rtdetrv1-s_oai_2026-02-03/rtdetr_r18vd_dec3_6x_coco_from_paddle.pt")

    results = model.train(data="/mnt/data/afarec/data/OpenAnimalImages/dataset.yaml",
                          epochs=100, imgsz=640, device="cuda", batch=16,
                          save=True, save_period=1,
                          name="rtdetrv1-s_oai_2026-02-03")
    # print(results)
    pass


def resume_training() -> None:
    model = YOLO("/mnt/data/afarec/code/runs/detect/rtdetrv1-l_oai_2026-02-02/weights/last.pt")
    results = model.train(resume=True)
    # print(results)


if __name__ == '__main__':
    train_model()
    # resume_training()