from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def str_to_timedelta(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, microseconds = seconds.split('.')
    return timedelta(
        hours=int(hours),
        minutes=int(minutes),
        seconds=int(seconds),
        microseconds=int(microseconds)
    )


cls_mapping = {
    'all': ['cat', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster',
            'hedgehog', 'javasparrow', 'parakeet', 'rabbit'],
    'bird': ['javasparrow', 'parakeet'],
    'small_animals': ['chinchilla', 'degus', 'ferret', 'guineapig', 'hamster', 'hedgehog', 'rabbit'],
    'cat': ['cat'],
    'dog': ['dog'],
}

reverse_cls_map = {}
for key, values in cls_mapping.items():
    for value in values:
        reverse_cls_map[value] = key


def auc_table_special():
    models = {
        'GhostV2-Arc': 'GhostFaceNets/work_dir/arcface_',
        'GhostV2-Cos': 'GhostFaceNets/work_dir/cosface_',
        'ArcFace-R34': 'ArcFace/work_dir/r34_arcface_',
        'ArcFace-R50': 'ArcFace/work_dir/r50_arcface_',
        'CosFace-R34': 'ArcFace/work_dir/r34_cosface_',
        'CosFace-R50': 'ArcFace/work_dir/r50_cosface_',
        'SphereFace20': 'SphereFace/work_dir_v2/20_',
        'SphereFace64': 'SphereFace/work_dir_v2/64_',
    }

    cls_names = ['bird', 'cat', 'dog', 'small_animals']
    s_general = []
    s_special = []


    work_dir = Path('/mnt/data/afarec/code/face_recognition/')
    for model_name, model_path in models.items():
        # General Models

        latency_path = work_dir / (model_path + 'all') / 'timing.txt'
        with latency_path.open('rt') as f:
            time_str = f'{str_to_timedelta(f.readline()).total_seconds()*1000:.1f}'
        r = {}
        res_path = work_dir / (model_path+'all') / 'verification.csv'
        df_cls = pd.read_csv(res_path)
        df_cls['species'] = df_cls['file1'].str.extract(r'([a-z]+[/\\]\d+)')[0].str.split('/', n=1, expand=True)[0]
        df_cls['cls'] = df_cls['species'].map(reverse_cls_map)

        for cls in cls_names:

            # filter df_cls for cls
            df_res = df_cls[df_cls['cls'] == cls]

            sims = np.array(df_res['sim'].tolist())
            sims = ((sims > 0) * sims).tolist()
            labs = df_res['label'].tolist()

            r[cls] = roc_auc_score(labs, sims) * 100

        r['all'] = roc_auc_score(df_cls['label'].tolist(), df_cls['sim'].tolist()) * 100

        cls_info_50 = ' & '.join([f'{r[l]:.2f}' for l in cls_names+['all']])
        s_general.append(f'{model_name} & {cls_info_50} & {time_str} \\\\ \\tabrowspace')

        # Special Models
        r = {}
        all_sims = []
        all_labels = []
        timings = []
        for cls in cls_names:
            res_path = work_dir / (model_path+cls) / 'verification.csv'
            df_res = pd.read_csv(res_path)

            sims = np.array(df_res['sim'].tolist())
            sims = ((sims > 0) * sims).tolist()
            labs = df_res['label'].tolist()

            all_sims.extend(sims)
            all_labels.extend(labs)

            r[cls] = roc_auc_score(labs, sims) * 100

            latency_path = work_dir / (model_path+cls) / 'timing.txt'
            with latency_path.open('rt') as f:
                timings.append(str_to_timedelta(f.readline()))

        r['all'] = roc_auc_score(all_labels, all_sims) * 100

        cls_info_50 = ' & '.join([f'{r[l]:.2f}' for l in cls_names+['all']])
        time_str = f'{(sum(timings, timedelta())/len(timings)).total_seconds()*1000:.1f}'
        s_special.append(f'{model_name} & {cls_info_50} & {time_str} \\\\ \\tabrowspace')


    for s in s_general:
        print(s)
    print('\\midrule')
    print('\\multicolumn{7}{c}{Specialised Models} \\\\')
    print('\\midrule')
    for s in s_special:
        print(s)


if __name__ == '__main__':
    auc_table_special()
