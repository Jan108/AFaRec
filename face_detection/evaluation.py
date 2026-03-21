from pathlib import Path

import fiftyone as fo
import numpy as np
import pandas as pd
from fiftyone import ViewField as F
from matplotlib import pyplot as plt
from matplotlib import colormaps
from sklearn.metrics import average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, \
    recall_score
from statsmodels.stats.inter_rater import fleiss_kappa

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

    for scrfd_type in ['2.5', '10', '34']:
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


def _ndarray_to_binary(arr: np.ndarray) -> np.ndarray:
    return (arr != np.sort(np.unique(arr))[0]).astype(int)


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


def convert_name(pred: str) -> tuple[str, str, str]:
    style = '-'
    match pred.split('_')[1]:
        case 'resnet18':
            color = '#1f77b4'
        case 'resnet34':
            color = '#ff7f0e'
        case 'mobilenetv2':
            color = '#2ca02c'
        case '25':
            color = '#1f77b4'
        case '10':
            color = '#ff7f0e'
        case '34':
            color = '#2ca02c'
        case 's':
            color = '#ff7f0e'
        case _:
            color = '#1f77b4'

    if pred.endswith('_all'):
        pred = pred.replace('_all', '')
        style = ':'
    elif pred.endswith('_pretrained'):
        pred = pred.replace('_pretrained', '')
        style = '--'
    elif pred.endswith('_combined'):
        pred = pred.replace('_combined', '')
        style = '-'
    if pred.startswith('retinaface_'):
        pred = pred[len('retinaface_'):]
        pred = pred.replace('resnet','R')
        pred = pred.replace('mobilenetv2','MNet')
    elif pred.startswith('scrfd_'):
        pred = pred[len('scrfd_'):]
        pred = pred.replace('25','2.5G')
        pred = pred.replace('10','10G')
        pred = pred.replace('34','34G')
    elif pred.startswith('yunet'):
        pred = pred[len('yunet'):]
        if pred.startswith('_s'):
            pred = pred.replace('_s','Small ')
        else:
            pred = 'Normal '+pred

    return pred, style, color


def presision_recall_curve():
    oafi_full = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/'))
    oafi = oafi_full.match_tags('test').match_tags('annotated').match_tags('no_face', bool=False)

    models = ([f'retinaface_{b}' for b in ['resnet18', 'resnet34', 'mobilenetv2']] +
              [f'scrfd_{t}' for t in ['2.5', '10', '34']] +
              ['yunet', 'yunet_s'])
    pred_fields = []
    for m in models:
        pred_fields += [f'{m}_pretrained', f'{m}_all', f'{m}_']
    # pred_fields = [f'{m}_pretrained' for m in models]
    # pred_fields += [f'{m}_all' for m in models]
    # pred_fields += [f'{m}_' for m in models]

    # pred_fields = ['retinaface_mobilenetv2_pretrained', 'scrfd_10_all', 'yunet_pretrained', 'yunet_all', 'yunet_', 'yunet_s_pretrained', 'yunet_s_all', 'yunet_s_']
    cls_names = ['bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']

    fig, [ax_retina, ax_scrfd, ax_yunet] = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
    thresholds = np.linspace(0, 1, 21)

    ax_retina.set_title("RetinaFace", fontsize=24)
    ax_scrfd.set_title("SCRFD", fontsize=24)
    ax_yunet.set_title("YuNet", fontsize=24)
    fig.suptitle("Recall-Confidence Curves", fontsize=24)

    for ax in [ax_retina, ax_scrfd, ax_yunet]:
        ax.grid(linestyle="--")
        ax.set_xlabel("Confidence Threshold", fontsize=20)
        ax.set_ylabel("Recall", fontsize=20)
        ax.tick_params(labelsize=20)

    legend_handles_retina = []
    legend_handles_scrfd = []
    legend_handles_yunet = []

    work_dir = Path('/mnt/data/afarec/code/face_detection/')
    for pred in pred_fields:
        if pred.startswith('retinaface_'):
            model = 'RetinaFace'
            ax_rc = ax_retina
            legend_handles = legend_handles_retina
        elif pred.startswith('scrfd_'):
            model = 'SCRFD'
            ax_rc = ax_scrfd
            legend_handles = legend_handles_scrfd
        elif pred.startswith('yunet_'):
            model = 'YuNet'
            ax_rc = ax_yunet
            legend_handles = legend_handles_yunet
        else:
            raise NotImplementedError(f'Unknown model: {pred}')

        if pred.endswith('_'):
            pred_base = pred
            pred = pred_base + 'combined'
            for cls in cls_names:
                res_path = work_dir / model / 'work_dir' / (pred_base + cls) / 'results'
                pred = pred.replace('.', '')
                import_prediction_from_yunet(oafi, res_path, pred, speedup=True)
        else:
            res_path = work_dir / model / 'work_dir' / pred / 'results'
            pred = pred.replace('.', '')
            import_prediction_from_yunet(oafi, res_path, pred, speedup=True)

        view = oafi.filter_labels(pred, F("confidence") > 0.05, only_matches=False)
        results = view.evaluate_detections(pred, gt_field="gt_face")

        ytrue = _ndarray_to_binary(results.ytrue)
        yconf = np.nan_to_num(results.confs.astype(float), 0)

        recalls = []
        for threshold in thresholds:
            y_pred = (yconf >= threshold).astype(int)
            recall = recall_score(ytrue, y_pred, zero_division=0)
            recalls.append(recall)
        label, linestyle, color = convert_name(pred)
        ax_rc.plot(thresholds, recalls, label=label, linestyle=linestyle, color=color)
        if linestyle == '-':
            legend_handles.append(
                plt.Line2D([0], [0], color=color, linestyle='-', linewidth=2, label=label)
            )
        ax_rc.legend(handles=legend_handles, fontsize=20)

        oafi_full.delete_sample_field(pred)

    plt.tight_layout()
    save_path = Path('/mnt/data/afarec/code/docs/figures/face_det_recall_conf.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def gen_tables():
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


def oafi_iaa():
    oafi_full = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OAFI_full/'))
    oafi_iaa = oafi_full.match_tags('iaa')

    field_a1 = 'IAA-jan'
    field_a2 = 'IAA-this'
    field_a3 = 'IAA-birgit'

    a1_cls = np.array([len(d) for d in oafi_iaa.values(field_a1+'.detections')])
    a2_cls = np.array([len(d) for d in oafi_iaa.values(field_a2+'.detections')])
    a3_cls = np.array([len(d) for d in oafi_iaa.values(field_a3+'.detections')])

    annotations = np.vstack((a1_cls, a2_cls, a3_cls))
    n_items = len(a1_cls)
    fleiss_data = np.zeros((n_items, 2))

    for i in range(n_items):
        counts = np.bincount(annotations[:, i], minlength=2)
        fleiss_data[i, :] = counts

    iaa_score = fleiss_kappa(fleiss_data)
    print(f'Calculated IAA Fleiss Kappa score for OAFI over {oafi_full.count()} samples: {iaa_score}')

    a1a2 = oafi_iaa.evaluate_detections(field_a1, gt_field=field_a2)
    a1a3 = oafi_iaa.evaluate_detections(field_a1, gt_field=field_a3)
    a2a3 = oafi_iaa.evaluate_detections(field_a2, gt_field=field_a3)

    ids = oafi_iaa.values(['id', f'{field_a1}.detections.id', f'{field_a2}.detections.id'])
    df_oafi_ids = pd.DataFrame({'sample_id': ids[0], field_a1: [i[0] if len(i) > 0 else '' for i in ids[1]],
                                field_a2: [i[0] if len(i) > 0 else '' for i in ids[2]]})

    df_a1a2 = pd.DataFrame({field_a1: a1a2.ypred_ids, 'a1a2_ious': a1a2.ious}).dropna()
    df_a1a3 = pd.DataFrame({field_a1: a1a3.ypred_ids, 'a1a3_ious': a1a3.ious}).dropna()
    df_a2a3 = pd.DataFrame({field_a2: a2a3.ypred_ids, 'a2a3_ious': a2a3.ious}).dropna()

    df_ious = pd.merge(df_oafi_ids, df_a1a2, on=field_a1, how='left')
    df_ious = pd.merge(df_ious, df_a1a3, on=field_a1, how='left')
    df_ious = pd.merge(df_ious, df_a2a3, on=field_a2, how='left')

    df_ious = df_ious.drop(columns=[field_a1, field_a2])
    df_ious['ious_avg'] = df_ious[['a1a2_ious', 'a1a3_ious', 'a2a3_ious']].mean(axis=1, skipna=True)

    a1a2_res = (df_ious['a1a2_ious'].mean(skipna=True), df_ious['a1a2_ious'].std(skipna=True))
    a1a3_res = (df_ious['a1a3_ious'].mean(skipna=True), df_ious['a1a3_ious'].std(skipna=True))
    a2a3_res = (df_ious['a2a3_ious'].mean(skipna=True), df_ious['a2a3_ious'].std(skipna=True))
    all_res = (df_ious['ious_avg'].mean(skipna=True), df_ious['ious_avg'].std(skipna=True))

    print(f'A1A2: {a1a2_res[0]}+-{a1a2_res[1]}')
    print(f'A1A3: {a1a3_res[0]}+-{a1a3_res[1]}')
    print(f'A2A3: {a2a3_res[0]}+-{a2a3_res[1]}')
    print(f'AVG: {all_res[0]}+-{all_res[1]}')
    print(f'{' & '.join([f'${a[0]*100:.2f}\\pm{a[1]*100:.2f}$' for a in [a1a2_res, a1a3_res, a2a3_res, all_res]])}')

if __name__ == '__main__':
    # import_retinaface()
    # import_scrfd()
    # import_yunet()

    # presision_recall_curve()
    # gen_tables()

    oafi_iaa()
