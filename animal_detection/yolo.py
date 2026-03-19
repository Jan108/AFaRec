import datetime

from tqdm import tqdm
from ultralytics import YOLO
import fiftyone as fo


RUN_DIR = '/mnt/data/afarec/code/runs/detect/'


class_mapping = {
    0: 'bird',
    1: 'cat',
    2: 'cat_like',
    3: 'dog',
    4: 'dog_like',
    5: 'horse_like',
    6: 'small_animals'
}

coco_class_mapping = {
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse_like', #horse
    18: 'horse_like', #sheep
    19: 'horse_like', #cow
}


def train_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26s.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-s_2026-02-19")


def train_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26m.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-m_2026-02-19")


def resume_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO(RUN_DIR+"yolo-m_2026-02-19/weights/last.pt").train(resume=True)


def resume_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO(RUN_DIR+"yolo-m_2026-02-19/weights/last.pt").train(resume=True)


def add_prediction_to_dataset(model_path: str, dataset: fo.Dataset, prediction_field: str,
                              coco_classes: bool = False, confidence: float = 0.25) -> datetime.timedelta:
    """
    Predict the detections using the given model for each sample in the dataset and save it under the prediction_field.
    :param model_path: Yolo model path for prediction
    :param dataset: loaded OAI dataset
    :param prediction_field: key where to store the predictions
    :param coco_classes: whether to use coco classes
    :param confidence: confidence threshold
    :return: average timedelta for one prediction
    """
    model = YOLO(model_path)
    times: list[datetime.timedelta] = []
    mapping = coco_class_mapping if coco_classes else class_mapping
    for sample in tqdm(dataset, desc=f'Inference for {prediction_field}:'):
        start_time = datetime.datetime.now()
        results = model(sample.filepath, verbose=False, conf=confidence)[0]
        end_time = datetime.datetime.now()
        times.append(end_time - start_time)

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalized xyxy
                conf = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                label = mapping.get(cls_idx, None)

                if label is not None:
                    detections.append(
                        fo.Detection(
                            label=label,
                            bounding_box=[x1, y1, x2 - x1, y2 - y1],  # [nx, ny, nw, nh]
                            confidence=conf,
                        )
                    )

        sample[prediction_field] = fo.Detections(detections=detections)
        sample.save()
    return sum(times, datetime.timedelta()) / len(times)
