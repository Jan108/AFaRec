from pathlib import Path

import fiftyone as fo
import numpy as np
from fiftyone import ViewField as F
from sklearn.metrics import average_precision_score
from supervision.metrics.mean_average_precision import MeanAveragePrecision, MeanAveragePrecisionResult
from tqdm import tqdm
import supervision as sv


from data.oafi import import_prediction_from_yunet
from data.utils import setup, load_yolo_dataset_from_disk

setup()

def import_retinaface():
    work_dir = Path('/mnt/data/afarec/code/face_detection/RetinaFace/work_dir')

    oafi = fo.load_dataset('OAFI_full')

    for backbone in ['resnet18', 'resnet34', 'mobilenetv2']:
        for cls in ["pretrained", "all", "bird", "cat", "cat_like", "dog", "dog_like", "horse_like", "small_animals"]:
            res_path = work_dir / f'retinaface_{backbone}_{cls}' / 'results'
            import_prediction_from_yunet(oafi, res_path, f'retinaface_{backbone}_{cls}', speedup=True)


def import_scrfd():
    work_dir = Path('/mnt/data/afarec/code/face_detection/SCRFD/work_dir')
    oafi = fo.load_dataset('OAFI_full')

    # for scrfd_type in ['2.5', '10', '34']:
    for scrfd_type in ['34']:
        for cls in ["pretrained", "all", "bird", "cat", "cat_like", "dog", "dog_like", "horse_like", "small_animals"]:
            res_path = work_dir / f'scrfd_{scrfd_type}_{cls}' / 'results'
            import_prediction_from_yunet(oafi, res_path, f'scrfd_{scrfd_type.replace('.','')}_{cls}', speedup=True)


def import_yunet():
    work_dir = Path('/mnt/data/afarec/code/face_detection/YuNet/work_dir')
    oafi = fo.load_dataset('OAFI_full')

    # for net in ['yunet', 'yunet_s']:
    for net in ['yunet']:
        for cls in ["pretrained", "all", "bird", "cat", "cat_like", "dog", "dog_like", "horse_like", "small_animals"]:
            res_path = work_dir / f'{net}_{cls}' / 'results'
            import_prediction_from_yunet(oafi, res_path, f'{net}_{cls}', speedup=True)


def _ndarray_to_binary(arr: np.ndarray) -> np.ndarray:
    return (arr != np.sort(np.unique(arr))[0]).astype(int)


def _to_binary(v: str) -> np.ndarray:
    return np.array([0]) if v.startswith('(') else np.array([1])


def convert_fiftyone_to_supervision(detection: fo.Detections) -> sv.Detections:
    result = []
    for det in detection.detections:
        nx, ny, nw, nh = det.bounding_box
        sv_det = sv.Detections(
            xyxy=np.array([[nx, ny, nx+nw, ny+nh]]),
            confidence=np.array([det.confidence]),
            class_id=_to_binary(det.label),
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


def create_table_map(pretrained: bool = True):
    oafi_full = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/'))
    oafi = oafi_full.match_tags('test').match_tags('annotated').match_tags('no_face', bool=False)
    # oafi = oai.take(250)

    models = ([f'retinaface_{b}' for b in ['resnet18', 'resnet34', 'mobilenetv2']] +
              [f'scrfd_{t}' for t in ['2.5', '10', '34']] +
              ['yunet', 'yunet_s'])

    if pretrained:
        pred_fields = [f'{m}_pretrained' for m in models]
    else:
        pred_fields = [f'{m}_all' for m in models]

    cls_names = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']
    s50 = []

    work_dir = Path('/mnt/data/afarec/code/face_detection/')
    for pred in pred_fields:
        if pred.startswith('retinaface_'):
            model = 'RetinaFace'
        elif pred.startswith('scrfd_'):
            model = 'SCRFD'
        elif pred.startswith('yunet_'):
            model = 'YuNet'
        else:
            raise NotImplementedError(f'Unknown model: {pred}')

        res_path = work_dir / model / 'work_dir' / pred / 'results'
        pred = pred.replace('.', '')
        import_prediction_from_yunet(oafi, res_path, pred, speedup=True)

        r = {}
        for cls in cls_names:
            cls_view = oafi.filter_labels('ground_truth', F('label').is_in([cls]))
            cls_view = cls_view.filter_labels(pred, F("confidence") > 0.05, only_matches=False)
            results = cls_view.evaluate_detections(pred, gt_field="gt_face")
            r[cls] = average_precision_score(
                y_true=_ndarray_to_binary(results.ytrue),
                y_score=_ndarray_to_binary(results.ypred)
            )

        view = oafi.filter_labels(pred, F("confidence") > 0.05, only_matches=False)
        results = view.evaluate_detections(pred, gt_field="gt_face")
        r['all'] = average_precision_score(
            y_true=_ndarray_to_binary(results.ytrue),
            y_score=_ndarray_to_binary(results.ypred)
        )

        cls_info_50 = ' & '.join([f'{r[l]*100:.2f}' for l in cls_names+['all']])
        s50.append(f'{pred} & {cls_info_50} & time \\\\ \\tabrowspace')
        print(s50[-1])

        oafi_full.delete_sample_field(pred)

    for s in s50:
        print(s)
    return s50


def create_table_map_classwise():
    oafi_full = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/'))
    oafi = oafi_full.match_tags('test').match_tags('annotated').match_tags('no_face', bool=False)
    # oafi = oai.take(250)

    models = ([f'retinaface_{b}' for b in ['resnet18', 'resnet34', 'mobilenetv2']] +
              [f'scrfd_{t}' for t in ['2.5', '10', '34']] +
              ['yunet', 'yunet_s'])

    cls_names = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']
    s50 = []

    work_dir = Path('/mnt/data/afarec/code/face_detection/')
    for model_name in models:

        pred_base = model_name + '_'
        pred = pred_base + 'combined'

        if pred.startswith('retinaface_'):
            model = 'RetinaFace'
        elif pred.startswith('scrfd_'):
            model = 'SCRFD'
        elif pred.startswith('yunet_'):
            model = 'YuNet'
        else:
            raise NotImplementedError(f'Unknown model: {pred}')

        r = {}
        for cls in cls_names:
            res_path = work_dir / model / 'work_dir' / (pred_base + cls) / 'results'
            pred = pred.replace('.', '')
            import_prediction_from_yunet(oafi, res_path, pred, speedup=True)

            cls_view = oafi.filter_labels('ground_truth', F('label').is_in([cls]))
            cls_view = cls_view.filter_labels(pred, F("confidence") > 0.05, only_matches=False)
            results = cls_view.evaluate_detections(pred, gt_field="gt_face")
            r[cls] = average_precision_score(
                y_true=_ndarray_to_binary(results.ytrue),
                y_score=_ndarray_to_binary(results.ypred)
            )

        view = oafi.filter_labels(pred, F("confidence") > 0.05, only_matches=False)
        results = view.evaluate_detections(pred, gt_field="gt_face")
        r['all'] = average_precision_score(
            y_true=_ndarray_to_binary(results.ytrue),
            y_score=_ndarray_to_binary(results.ypred)
        )

        cls_info_50 = ' & '.join([f'{r[l]*100:.2f}' for l in cls_names+['all']])
        s50.append(f'{pred} & {cls_info_50} & time \\\\ \\tabrowspace')
        print(s50[-1])

        oafi_full.delete_sample_field(pred)

    for s in s50:
        print(s)
    return s50


def calc_ap_conv_curve():
    oafi_full = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/'))
    oafi = oafi_full.match_tags('test').match_tags('annotated').match_tags('no_face', bool=False)
    cls_names = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']

    models = [f'retinaface_{b}' for b in ['resnet18', 'resnet34', 'mobilenetv2']]

    pred_fields = [f'{m}_all' for m in models]


    work_dir = Path('/mnt/data/afarec/code/face_detection/')
    scores = {}
    for pred in pred_fields:

        if pred.startswith('retinaface_'):
            model = 'RetinaFace'
        elif pred.startswith('scrfd_'):
            model = 'SCRFD'
        elif pred.startswith('yunet_'):
            model = 'YuNet'
        else:
            raise NotImplementedError(f'Unknown model: {pred}')

        res_path = work_dir / model / 'work_dir' / pred / 'results'
        pred = pred.replace('.', '')
        import_prediction_from_yunet(oafi, res_path, pred, speedup=True)

        res = {}
        for conf in [0,0.25,0.5,0.75,0.99]:
            r = {}
            for cls in cls_names:
                cls_view = oafi.filter_labels('ground_truth', F('label').is_in([cls]))
                cls_view = cls_view.filter_labels(pred, F("confidence") > conf, only_matches=False)
                results = cls_view.evaluate_detections(pred, gt_field="ground_truth", classwise=False)
                r[cls] = average_precision_score(
                    y_true=_ndarray_to_binary(results.ytrue),
                    y_score=_ndarray_to_binary(results.ypred)
                )


            view = oafi.filter_labels(pred, F("confidence") > conf, only_matches=False)
            results = view.evaluate_detections(pred, gt_field="ground_truth", classwise=False)
            r['all'] = average_precision_score(
                y_true=_ndarray_to_binary(results.ytrue),
                y_score=_ndarray_to_binary(results.ypred)
            )
            res[conf] = r
        scores[pred] = res

        oafi_full.delete_sample_field(pred)

    print(scores)
    pass


if __name__ == '__main__':
    # import_retinaface()
    # import_scrfd()
    # import_yunet()


    print('Table pretrained')
    s_pre = create_table_map(True)
    print('Table generell')
    s_ft = create_table_map(False)
    print('Table classwise')
    s_cls = create_table_map_classwise()


    for s in s_pre:
        print(s)
    print('\\midrule')
    print('\\multicolumn{10}{c}{Generell Models} \\\\')
    print('\\midrule')
    for s in s_ft:
        print(s)
    print('\\midrule')
    print('\\multicolumn{10}{c}{Specialised Models} \\\\')
    print('\\midrule')
    for s in s_cls:
        print(s)