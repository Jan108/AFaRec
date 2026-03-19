import datetime
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rfdetr import RFDETRSmall, RFDETRMedium
from supervision.dataset.core import DetectionDataset
from supervision.metrics import MeanAveragePrecision
import supervision as sv
import fiftyone as fo
from tqdm import tqdm

from data.utils import convert_xyxy_to_xywhn


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
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse_like', #horse
    20: 'horse_like', #sheep
    21: 'horse_like', #cow
}


def train_rfdetr_s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    RFDETRSmall().train(
        dataset_dir=dataset_dir+'OpenAnimalImages_RF-DETR', dataset_file='yolo',
        epochs=50, batch_size=4, grad_accum_steps=4, checkpoint_interval=1, device="cuda",
        output_dir=dataset_dir+'runs/rfdetr/small-2026-02-19'
    )


def resume_rfdetr_s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    RFDETRSmall().train(
        dataset_dir=dataset_dir+'OpenAnimalImages_RF-DETR', dataset_file='yolo',
        epochs=50, batch_size=4, grad_accum_steps=4, checkpoint_interval=1, device="cuda",
        output_dir=dataset_dir+'runs/rfdetr/small-2026-02-19',
        resume='????????????????',
    )


def add_prediction_to_dataset(model: RFDETRSmall | RFDETRMedium, dataset: fo.Dataset, prediction_field: str,
                              coco_classes: bool = False, confidence: float = 0.5) -> datetime.timedelta:
    """
    Predict the detections using the given model for each sample in the dataset and save it under the prediction_field.
    :param model: RFDETR model for prediction
    :param dataset: loaded OAI dataset
    :param prediction_field: key where to store the predictions
    :param coco_classes: whether to use coco classes
    :param confidence: confidence threshold
    :return: average timedelta for one prediction
    """
    times: list[datetime.timedelta] = []
    model.optimize_for_inference()
    mapping = coco_class_mapping if coco_classes else class_mapping
    for sample in tqdm(dataset, desc=f'Inference for {prediction_field}:'):
        start_time = datetime.datetime.now()
        img = Image.open(sample.filepath).convert('RGB')
        results: sv.Detections = model.predict(img, threshold=confidence)
        end_time = datetime.datetime.now()
        times.append(end_time - start_time)

        detections = []
        for xyxy, _, conf, class_id, _, _ in results:
            label = mapping.get(class_id, None)
            if label is not None:
                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=convert_xyxy_to_xywhn(xyxy, img.width, img.height),
                        confidence=conf,
                    )
                )

        sample[prediction_field] = fo.Detections(detections=detections)
        sample.save()
    return sum(times, datetime.timedelta()) / len(times)


def eval_model(model: RFDETRSmall | RFDETRMedium, dataset_path: str = '/mnt/data/afarec/data/OpenAnimalImages_RF-DETR/') -> None:
    dataset = DetectionDataset.from_yolo(
        images_directory_path=dataset_path+'test/images',
        annotations_directory_path=dataset_path+'test/labels',
        data_yaml_path=dataset_path+'data.yaml',
    )
    predictions = model.predict(images=dataset.image_paths)

    map_metric = MeanAveragePrecision()
    map_metric.update(predictions,  [dataset.annotations[img] for img in dataset.image_paths])
    map_results = map_metric.compute()
    print('RFDETR results:')
    print(map_results)


def parse_eval(log_file: Path) -> list[dict[str, Any]]:
    eval_data = []
    with log_file.open('r') as f:
        for line in f:
            eval_data.append(json.loads(line))
    return eval_data
