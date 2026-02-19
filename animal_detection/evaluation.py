from pathlib import Path

import fiftyone as fo
import supervision as sv
from rfdetr import RFDETRSmall
from supervision.metrics.mean_average_precision import MeanAveragePrecision, MeanAveragePrecisionResult
import numpy as np

from data.creation import load_yolo_dataset_from_disk
import yolo as yolo_stuff
import rf_detr as rfdetr_stuff


class_mapping = {
    'bird': 0,
    'cat': 1,
    'cat_like': 2,
    'dog': 3,
    'dog_like': 4,
    'horse_like': 5,
    'small_animals': 6
}


def convert_fiftyone_to_supervision(detection: fo.Detections) -> sv.Detections:
    result = []
    for det in detection.detections:
        nx, ny, nw, nh = det.bounding_box
        sv_det = sv.Detections(
            xyxy=np.array([[nx, ny, nx+nw, ny+nh]]),
            confidence=np.array([det.confidence]),
            class_id=np.array([class_mapping[det.label]]),
            tracker_id=np.array([det.id]),
        )
        result.append(sv_det)
    return sv.Detections.merge(result)


def calc_mAP(dataset: fo.Dataset, prediction_field: str, ground_truth_field: str = 'ground_truth') -> MeanAveragePrecisionResult:
    map_metric = MeanAveragePrecision()
    for sample in dataset:
        prediction = sample[prediction_field]
        ground_truth = sample[ground_truth_field]
        map_metric.update(
            convert_fiftyone_to_supervision(prediction),
            convert_fiftyone_to_supervision(ground_truth)
        )
    return map_metric.compute()


def run_eval_yolo(oai_Dataset: fo.Dataset):
    print('Eval of Yolo26s 2026-02-13 best weights')
    yolo_stuff.add_prediction_to_dataset(
        model_path='/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-s_oai_2026-02-13_v3/weights/best.pt',
        dataset=oai_Dataset,
        prediction_field="Yolo26S_02_13_best",
    )
    print(calc_mAP(oai_Dataset, prediction_field='Yolo26S_02_13_best'))
    print(f'{'=' * 150}')

    print('Eval of Yolo26m 2026-02-13 best weights')
    yolo_stuff.add_prediction_to_dataset(
        model_path='/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-m_oai_2026-02-13_v4/weights/best.pt',
        dataset=oai_Dataset,
        prediction_field="Yolo26M_02_13_best",
    )
    print(calc_mAP(oai_Dataset, prediction_field='Yolo26M_02_13_best'))
    print(f'{'=' * 150}')

    # print('Eval of Yolo26s')
    # yolo_stuff.add_prediction_to_dataset(
    #     model_path='yolo26s.pt',
    #     dataset=oai_Dataset,
    #     prediction_field="Yolo26S_baseline",
    # )
    # print(calc_mAP(oai_Dataset, prediction_field='Yolo26S_baseline'))
    # print(f'{'='*150}')
    #
    # print('Eval of Yolo26m')
    # yolo_stuff.add_prediction_to_dataset(
    #     model_path='yolo26m.pt',
    #     dataset=oai_Dataset,
    #     prediction_field="Yolo26M_baseline",
    # )
    # print(calc_mAP(oai_Dataset, prediction_field='Yolo26M_baseline'))
    # print(f'{'='*150}')


def run_eval_rfdetr(oai_Dataset: fo.Dataset):
    print('Eval of RF-Detr 2026-02-04 epoch 27')
    rfdetr_stuff.add_prediction_to_dataset(
        model=RFDETRSmall(pretrain_weights='/mnt/data/afarec/code/runs/rfdetr/2026-02-04/checkpoint0027.pth'),
        dataset=oai_Dataset,
        prediction_field="RFDetrS_02_04_e27",
    )
    print(calc_mAP(oai_Dataset, prediction_field='RFDetrS_02_04_e27'))
    print(f'{'=' * 150}')


if __name__ == "__main__":
    dataset = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OpenAnimalImages/')).match_tags('test')

    run_eval_yolo(dataset)
    run_eval_rfdetr(dataset)
