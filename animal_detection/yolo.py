from tqdm import tqdm
from ultralytics import YOLO
import fiftyone as fo


RUN_DIR = '/mnt/data/afarec/code/runs/detect/'


def train_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26s.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=64, save=True, save_period=1,
                             name="yolo-s_2026-02-19")


def train_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26m.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=64, save=True, save_period=1,
                             name="yolo-m_2026-02-19")


def resume_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO(RUN_DIR+"yolo-m_2026-02-19/weights/last.pt").train(resume=True)


def resume_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO(RUN_DIR+"yolo-m_2026-02-19/weights/last.pt").train(resume=True)


def add_prediction_to_dataset(model_path: str, dataset: fo.Dataset, prediction_field: str,
                              coco_classes: bool = False, confidence: float = 0.25) -> None:
    """
    Predict the detections using the given model for each sample in the dataset and save it under the prediction_field.
    :param model_path: Yolo model path for prediction
    :param dataset: loaded OAI dataset
    :param prediction_field: key where to store the predictions
    :param coco_classes: whether to use coco classes
    :param confidence: confidence threshold
    :return: None
    """
    model = YOLO(model_path)
    for sample in tqdm(dataset, desc=f'Inference for {prediction_field}:'):
        results = model(sample.filepath, verbose=False, conf=confidence)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalized xyxy
                conf = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                label = results.names[cls_idx] if cls_idx in results.names else f"class_{cls_idx}"

                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=[x1, y1, x2 - x1, y2 - y1],  # [nx, ny, nw, nh]
                        confidence=conf,
                    )
                )

        sample[prediction_field] = fo.Detections(detections=detections)
        sample.save()
