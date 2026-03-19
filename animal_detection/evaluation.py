from pathlib import Path

import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from fiftyone import ViewField as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rfdetr import RFDETRSmall, RFDETRMedium
from sklearn.metrics import ConfusionMatrixDisplay
from supervision.metrics.mean_average_precision import MeanAveragePrecision, MeanAveragePrecisionResult
from tqdm import tqdm

import rf_detr as rfdetr_stuff
import yolo as yolo_stuff
from data import utils

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
    for sample in tqdm(dataset, desc=f'{prediction_field} mAP'):
        prediction = sample[prediction_field]
        ground_truth = sample[ground_truth_field]
        map_metric.update(
            convert_fiftyone_to_supervision(prediction),
            convert_fiftyone_to_supervision(ground_truth)
        )
    return map_metric.compute()


def calc_mAP_per_cls(dataset: fo.Dataset, prediction_field: str,
                     ground_truth_field: str = 'ground_truth', cls_list: list[str] = None
                     ) -> dict[str, MeanAveragePrecisionResult]:
    res = {}
    if cls_list is None:
        cls_list = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']
    for cls in cls_list:
        res[cls] = calc_mAP(dataset.filter_labels(ground_truth_field, F('label').is_in([cls])).filter_labels(prediction_field, F('label').is_in([cls])),
                            prediction_field, ground_truth_field)
    return res

def create_table_map():
    oai = utils.load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OpenAnimalImages/')).match_tags('test')
    # oai = oai.take(250)

    add_yolo_to_dataset(oai)
    add_rfdetr_to_dataset(oai)

    pred_fields = [
        'Yolo26S_02_13_best', 'Yolo26M_02_13_best', 'Yolo26S_baseline', 'Yolo26M_baseline',
        'RFDetrS_02_22_best', 'RFDetrM_02_22_best', 'RFDetrS_baseline', 'RFDetrM_baseline'
    ]

    results = {}
    cls_names = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']
    coco_cls = ['bird', 'cat', 'dog', 'horse_like']
    s50 = []
    s95 = []
    for pred in pred_fields:
        if pred.endswith('baseline'):
            r = calc_mAP_per_cls(oai, pred, cls_list=coco_cls)
            r['all'] = calc_mAP(oai.filter_labels('ground_truth', F('label').is_in(coco_cls)), pred)
        else:
            r = calc_mAP_per_cls(oai, pred)
            r['all'] = calc_mAP(oai, pred)
        results[pred] = r

        cls_info_50 = ' & '.join([f'{r[l].map50*100:.2f}' if l in r else '--' for l in cls_names+['all']])
        cls_info_95 = ' & '.join([f'{r[l].map50_95*100:.2f}' if l in r else '--' for l in cls_names+['all']])
        s50.append(f'{pred[0:7]} & \\{'x' if pred.endswith('baseline') else 'c'}mark & {cls_info_50} & time \\\\ \\tabrowspace')
        s95.append(f'{pred[0:7]} & \\{'x' if pred.endswith('baseline') else 'c'}mark & {cls_info_95} & time \\\\ \\tabrowspace')

    for s in s50:
        print(s)
    print('\\midrule')
    print('\\multicolumn{11}{c}{mAP50:95 (\\%)} \\\\')
    print('\\midrule')
    for s in s95:
        print(s)

    print(results)


def create_conf_plot():
    oai = utils.load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OpenAnimalImages/')).match_tags('test')
    # oai = oai.take(250)

    pred_fields = [
        'Yolo26S_02_13_best', 'Yolo26M_02_13_best', # 'Yolo26S_baseline', 'Yolo26M_baseline',
        'RFDetrS_02_22_best', 'RFDetrM_02_22_best', # 'RFDetrS_baseline', 'RFDetrM_baseline'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    axes = axes.flatten()

    display_labels = ['Background', 'Bird', 'Cat', 'Cat-like', 'Dog', 'Dog-like', 'Horse-like', 'Small Animal']
    display_labels = ['BG', 'Bird', 'Cat', 'CatL', 'Dog', 'DogL', 'Horse', 'SmAn']
    titles = ['Yolo26-S', 'Yolo26-M', 'RFDetr-S', 'RFDetr-M']
    for i, (pred, ax) in enumerate(zip(pred_fields, axes)):
        results = oai.evaluate_detections(pred, gt_field="ground_truth")

        disp = ConfusionMatrixDisplay.from_predictions(
            results.ytrue,
            results.ypred,
            # normalize='true',
            cmap='Blues',
            xticks_rotation=45,
            display_labels=display_labels,
            # values_format='.2f',
            ax=ax,
            colorbar=False,
        )

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('Predicted', fontsize=16)
        ax.set_ylabel('True', fontsize=16)

        ax.set_title(titles[i], fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(disp.im_, cax=cax)
        cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.4, hspace=0.3)
    plt.tight_layout(pad=0.6)
    save_path = Path('/mnt/data/afarec/code/docs/figures/obj_det_conf_unorm.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def add_yolo_to_dataset(dataset: fo.Dataset):
    print('Eval of Yolo26s 2026-02-13 best weights')
    y26s_delta = yolo_stuff.add_prediction_to_dataset(
        model_path='/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-s_oai_2026-02-13_v3/weights/best.pt',
        dataset=dataset,
        prediction_field="Yolo26S_02_13_best",
        confidence=0.05
    )
    print(f'Took {y26s_delta} for prediction per image')

    print('Eval of Yolo26m 2026-02-13 best weights')
    y26m_delta = yolo_stuff.add_prediction_to_dataset(
        model_path='/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-m_oai_2026-02-13_v4/weights/best.pt',
        dataset=dataset,
        prediction_field="Yolo26M_02_13_best",
        confidence=0.05
    )
    print(f'Took {y26m_delta} for prediction per image')

    print('Eval of Yolo26s')
    y26s_pre_delta = yolo_stuff.add_prediction_to_dataset(
        model_path='yolo26s.pt',
        dataset=dataset,
        prediction_field="Yolo26S_baseline",
        confidence=0.05,
        coco_classes=True,
    )
    print(f'Took {y26s_pre_delta} for prediction per image')

    print('Eval of Yolo26m')
    y26m_pre_delta = yolo_stuff.add_prediction_to_dataset(
        model_path='yolo26m.pt',
        dataset=dataset,
        prediction_field="Yolo26M_baseline",
        confidence=0.05,
        coco_classes=True,
    )
    print(f'Took {y26m_pre_delta} for prediction per image')

    print(f'All times: {y26s_delta}, {y26m_delta}, {y26s_pre_delta}, {y26m_pre_delta}')
    # All times: 0:00:00.014727, 0:00:00.017649, 0:00:00.014386, 0:00:00.017635


def add_rfdetr_to_dataset(dataset: fo.Dataset):
    print('Eval of RF-Detr small 2026-02-22 best')
    s_delta = rfdetr_stuff.add_prediction_to_dataset(
        model=RFDETRSmall(pretrain_weights='/mnt/data/afarec/code/runs/mlserv2/runs/rfdetr/small-2026-02-22-2/checkpoint_best_total.pth'),
        dataset=dataset,
        prediction_field="RFDetrS_02_22_best",
        confidence=0.05
    )
    print(f'Took {s_delta} for prediction per image')

    print('Eval of RF-Detr medium 2026-02-22 best')
    m_delta = rfdetr_stuff.add_prediction_to_dataset(
        model=RFDETRMedium(pretrain_weights='/mnt/data/afarec/code/runs/mlserv2/runs/rfdetr/medium-2026-02-22-2/checkpoint_best_total.pth'),
        dataset=dataset,
        prediction_field="RFDetrM_02_22_best",
        confidence=0.05
    )
    print(f'Took {m_delta} for prediction per image')

    print('Eval of RF-Detr small baseline')
    s_pre_delta = rfdetr_stuff.add_prediction_to_dataset(
        model=RFDETRSmall(),
        dataset=dataset,
        prediction_field="RFDetrS_baseline",
        confidence=0.05,
        coco_classes=True,
    )
    print(f'Took {s_pre_delta} for prediction per image')

    print('Eval of RF-Detr medium baseline')
    m_pre_delta = rfdetr_stuff.add_prediction_to_dataset(
        model=RFDETRMedium(),
        dataset=dataset,
        prediction_field="RFDetrM_baseline",
        confidence=0.05,
        coco_classes=True,
    )
    print(f'Took {m_pre_delta} for prediction per image')

    print(f'All times: {s_delta}, {m_delta}, {s_pre_delta}, {m_pre_delta}')
    # All times: 0:00:00.028628, 0:00:00.033064, 0:00:00.028328, 0:00:00.031997


if __name__ == "__main__":
    # create_table_map()
    create_conf_plot()