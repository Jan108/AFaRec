from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from sklearn.metrics import average_precision_score

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

    for net in ['yunet', 'yunet_s']:
        for cls in ["pretrained", "all", "bird", "cat", "cat_like", "dog", "dog_like", "horse_like", "small_animals"]:
            res_path = work_dir / f'{net}_{cls}' / 'results'
            import_prediction_from_yunet(oafi, res_path, f'{net}_{cls}', speedup=True)


def create_table_map(pretrained: bool = True):
    oafi = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/')).match_tags('test')
    oafi = oafi.match_tags('annotated').match_tags('no_face', bool=False)
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
    for pred in pred_fields:
        r = {}
        for cls in cls_names:
            cls_view = oafi.filter_labels('ground_truth', F('label').is_in([cls]))
            results = cls_view.evaluate_detections(pred, gt_field="ground_truth")
            r[cls] = average_precision_score(results.ytrue, results.ypred)

        results = oafi.evaluate_detections(pred, gt_field="ground_truth")
        r['all'] = average_precision_score(results.ytrue, results.ypred)

        cls_info_50 = ' & '.join([f'{r[l]*100:.2f}' for l in cls_names+['all']])
        s50.append(f'{pred} & {cls_info_50} & time \\\\ \\tabrowspace')

    for s in s50:
        print(s)


if __name__ == '__main__':
    # import_retinaface()
    import_scrfd()
    import_yunet()

    print('Tabel pretrained')
    create_table_map(True)
    print('Tabel generell')
    create_table_map(False)