from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


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


def auc_table_verfication():
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
            time_str = f'{str_to_timedelta(f.readline().strip()).total_seconds()*1000:.1f}'
        r = {}
        res_path = work_dir / (model_path+'all') / 'verification.csv'
        df_cls = pd.read_csv(res_path)
        df_cls['species'] = df_cls['file1'].str.extract(r'([a-z]+[/\\]\d+)')[0].str.split('/', n=1, expand=True)[0]
        df_cls['cls'] = df_cls['species'].map(reverse_cls_map)

        all_sims = []
        all_labels = []
        for cls in cls_names:

            # filter df_cls for cls
            df_res = df_cls[df_cls['cls'] == cls]

            sims = np.array(df_res['sim'].tolist())
            sims = ((sims > 0) * sims).tolist()
            labs = df_res['label'].tolist()
            all_sims.extend(sims)
            all_labels.extend(labs)

            r[cls] = roc_auc_score(labs, sims) * 100

        r['all'] = roc_auc_score(all_labels, all_sims) * 100

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
                timings.append(str_to_timedelta(f.readline().strip()))

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


def plot_dist():
    work_dir = Path('/mnt/data/afarec/code/face_recognition/')
    models = {
        '(a) GhostV2-Arc': 'GhostFaceNets/work_dir/arcface_',
        # 'GhostV2-Cos': 'GhostFaceNets/work_dir/cosface_',
        # 'ArcFace-R34': 'ArcFace/work_dir/r34_arcface_',
        '(b) ArcFace-R50': 'ArcFace/work_dir/r50_arcface_',
        # 'CosFace-R34': 'ArcFace/work_dir/r34_cosface_',
        # 'CosFace-R50': 'ArcFace/work_dir/r50_cosface_',
        '(c) SphereFace20': 'SphereFace/work_dir_v2/20_',
        # 'SphereFace64': 'SphereFace/work_dir_v2/64_',
    }

    sns.set_style(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    # fig.suptitle("Distribution of similarity scores for generalized models", fontsize=24)

    for (model_name, model_path), ax in zip(models.items(), axes):
        res_path = work_dir / (model_path + 'all') / 'verification.csv'
        df_cls = pd.read_csv(res_path)
        df_cls['species'] = df_cls['file1'].str.extract(r'([a-z]+[/\\]\d+)')[0].str.split('/', n=1, expand=True)[0]
        df_cls['cls'] = df_cls['species'].map(reverse_cls_map)

        sns.kdeplot(data=df_cls, x='sim', hue='label', common_norm=False, hue_order=[1, 0],
                palette={0: '#ff7f0e', 1: '#2ca02c'}, ax=ax, fill=True, alpha=0.3)

        ax.set_title(model_name, fontsize=24)
        ax.set_xlabel('Similarity Score', fontsize=24)
        ax.set_ylabel('Density', fontsize=24)
        ax.legend(labels=['neg.', 'pos.'], fontsize=24)
        ax.tick_params(labelsize=24)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    save_path = Path('/mnt/data/afarec/code/docs/figures/face_rec_sim_dist.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def table_acc_identification():
    models = {
        'GhostV2-Arc': ('GhostFaceNets/work_dir/arcface_', 0.5510204081632653, 0.1),
        'GhostV2-Cos': ('GhostFaceNets/work_dir/cosface_', 0.0, 0.1),
        'ArcFace-R34': ('ArcFace/work_dir/r34_arcface_', 0.3469387755102041, 0.1),
        'ArcFace-R50': ('ArcFace/work_dir/r50_arcface_', 0.3061224489795918, 0.1),
        'CosFace-R34': ('ArcFace/work_dir/r34_cosface_', 0.32653061224489793, 0.1),
        'CosFace-R50': ('ArcFace/work_dir/r50_cosface_', 0.32653061224489793, 0.1),
        # 'SphereFace20': ('SphereFace/work_dir/20_', 0.1, 0.1),
        # 'SphereFace64': ('SphereFace/work_dir/64_', 0.1, 0.1),
    }

    cls_names = ['bird', 'cat', 'dog', 'small_animals']
    s_general = []
    s_special = []

    work_dir = Path('/mnt/data/afarec/code/face_recognition/')
    for model_name, (model_path, t_gen, t_spe) in models.items():
        # General Models
        latency_path = work_dir / (model_path + 'all') / 'timing.txt'
        with latency_path.open('rt') as f:
            f.readline()
            time_str = f'{str_to_timedelta(f.readline().strip()).total_seconds() * 1000:.1f}'

        r = {}
        list_df = []
        for cls in cls_names:
            res_path = work_dir / (model_path + 'all') / f'identification_{cls}.csv'

            df_ident = pd.read_csv(res_path)
            df_true = pd.read_csv(f'/mnt/data/afarec/data/PetFace/split/{cls}/identification_label.csv')
            df_true.rename(columns={'individual': 'test_label'}, inplace=True)

            df_merge = pd.merge(df_ident, df_true, on='test_label', how='left')
            r[cls] = calc_acc_identification(t_gen, 1, df_merge)
            list_df.append(df_merge)
        df_merge = pd.concat(list_df, axis='index')
        r['tpir'], r['fpir'], r['acc'] = calc_tpir_fpir_acc(t_gen, df_merge)

        cls_info_gen = ' & '.join([f'{r[l]*100:.2f}' for l in cls_names + ['acc', 'tpir', 'fpir']])
        s_general.append(f'{model_name} & {cls_info_gen} & {time_str} \\\\ \\tabrowspace')

        # Special Models
        r = {}
        list_df  = []
        timings = []
        for cls in cls_names:
            res_path = work_dir / (model_path + cls) / f'identification.csv'

            df_ident = pd.read_csv(res_path)
            df_true = pd.read_csv(f'/mnt/data/afarec/data/PetFace/split/{cls}/identification_label.csv')
            df_true.rename(columns={'individual': 'test_label'}, inplace=True)

            df_merge = pd.merge(df_ident, df_true, on='test_label', how='left')
            r[cls] = calc_acc_identification(t_spe, 1, df_merge)
            list_df.append(df_merge)

            latency_path = work_dir / (model_path + cls) / 'timing.txt'
            with latency_path.open('rt') as f:
                f.readline()
                timings.append(str_to_timedelta(f.readline().strip()))

        df_merge = pd.concat(list_df, axis='index')
        r['tpir'], r['fpir'], r['acc'] = calc_tpir_fpir_acc(t_spe, df_merge)

        cls_info_spe = ' & '.join([f'{r[l]*100:.2f}' for l in cls_names + ['acc', 'tpir', 'fpir']])
        time_str = f'{(sum(timings, timedelta()) / len(timings)).total_seconds() * 1000:.1f}'
        s_special.append(f'{model_name} & {cls_info_spe} & {time_str} \\\\ \\tabrowspace')

    for s in s_general:
        print(s)
    print('\\midrule')
    print('\\multicolumn{9}{c}{Specialized Models} \\\\')
    print('\\midrule')
    for s in s_special:
        print(s)


def calc_tpir_fpir_acc(t: float, df: pd.DataFrame) -> tuple[float, float, float]:
    df.loc[:, 'pred_label'] = df.loc[:, 'predicted_label_0']
    df.loc[df['similarity_0'] < t, 'pred_label'] = -1

    known = df['true_label'] != -1
    unknown = df['true_label'] == -1

    known_sum = known.sum()
    unknown_sum = unknown.sum()

    known_correct = ((df['true_label'] == df['pred_label']) & known).sum()
    unknown_wrong = ((df['true_label'] != df['pred_label']) & unknown).sum()
    correct = (df['true_label'] == df['pred_label']).sum()

    tpir = known_correct / known_sum if known_sum > 0 else 0
    fpir = unknown_wrong / unknown_sum if unknown_sum > 0 else 0
    acc = correct / (known_sum + unknown_sum)
    return tpir, fpir, acc


def calc_acc_identification(t: float, k: int, df_merge: pd.DataFrame) -> float:
    if not k in [1, 3, 5]:
        raise ValueError(f'k needs to be one of [1, 3, 5], not {k}')
    if t > 1 or t < -1:
        raise ValueError(f't needs to be between 1 and -1, not {t}')


    for i in range(k):
        df_merge[f't_label_{i}'] = df_merge.loc[:, f'predicted_label_{i}']
        df_merge.loc[(df_merge[f'similarity_{i}'] <= t), f't_label_{i}'] = -1
    correct = df_merge.apply(
        lambda row: row['true_label'] in row[[f't_label_{i}' for i in range(k)]].values,
        axis=1
    ).sum()
    return correct/len(df_merge)


def plot_radar_topk():
    models = {
        # 'GhostV2-Arc': ('GhostFaceNets/work_dir/arcface_', 0.5510204081632653),
        # 'GhostV2-Cos': ('GhostFaceNets/work_dir/cosface_', 0.0),
        'ArcFace-R34': ('ArcFace/work_dir/r34_arcface_', 0.3469387755102041),
        'ArcFace-R50': ('ArcFace/work_dir/r50_arcface_', 0.3061224489795918),
        'CosFace-R34': ('ArcFace/work_dir/r34_cosface_', 0.32653061224489793),
        'CosFace-R50': ('ArcFace/work_dir/r50_cosface_', 0.32653061224489793),
        # 'SphereFace20': ('SphereFace/work_dir/20_', 0.1),
        # 'SphereFace64': ('SphereFace/work_dir/64_', 0.1),
    }
    work_dir = Path('/mnt/data/afarec/code/face_recognition/')

    cls_labels = ['Bird', 'Cat', 'Dog', 'Small Animal']
    cls_names = ['bird', 'cat', 'dog', 'small_animals']

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5), subplot_kw={'polar': True})

    for ax, (model_name, (model_path, t)) in zip(axes, models.items()):

        ax.set_title(model_name, size=24)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cls_labels, color='grey', size=20)
        ax.set_rlabel_position(30)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for k, color in zip([1, 3, 5], colors):
            values = []
            for cls in cls_names:
                res_path = work_dir / (model_path + 'all') / f'identification_{cls}.csv'

                df_ident = pd.read_csv(res_path)
                df_true = pd.read_csv(f'/mnt/data/afarec/data/PetFace/split/{cls}/identification_label.csv')
                df_true.rename(columns={'individual': 'test_label'}, inplace=True)
                df_merge = pd.merge(df_ident, df_true, on='test_label', how='left')
                values.append(calc_acc_identification(t, k, df_merge)*100)
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=f'k={k}')
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0), fontsize=24)
        ax.tick_params(labelsize=18)

    plt.tight_layout()
    save_path = Path('/mnt/data/afarec/code/docs/figures/face_rec_radar_topk.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_acc_threshold(n_threshold: int = 10, generalized: bool = True):
    models = {
        'GhostV2-Arc': 'GhostFaceNets/work_dir/arcface_',
        'GhostV2-Cos': 'GhostFaceNets/work_dir/cosface_',
        'ArcFace-R34': 'ArcFace/work_dir/r34_arcface_',
        'ArcFace-R50': 'ArcFace/work_dir/r50_arcface_',
        'CosFace-R34': 'ArcFace/work_dir/r34_cosface_',
        'CosFace-R50': 'ArcFace/work_dir/r50_cosface_',
        'CosFace-R51': 'ArcFace/work_dir/r50_cosface_',
        'CosFace-R52': 'ArcFace/work_dir/r50_cosface_',
        # 'SphereFace20': 'SphereFace/work_dir/20_',
        # 'SphereFace64': 'SphereFace/work_dir/64_',
    }
    work_dir = Path('/mnt/data/afarec/code/face_recognition/')
    cls_names = ['bird', 'cat', 'dog', 'small_animals']
    thresholds = np.linspace(0, 1, n_threshold, endpoint=True).tolist()

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = iter(plt.get_cmap('tab10').colors)
    legend_handles_in = [
        ax.plot([0], [0], color='gray', linestyle='--', label='TPIR')[0],
        ax.plot([0], [0], color='gray', linestyle='-', label='FPIR')[0],
        ax.plot([0], [0], color='gray', linestyle=':', label='Acc.')[0],
    ]
    legend_handles = []

    # Plot each curve
    for model_name, model_path in models.items():
        tpirs = []
        fpirs = []
        accs = []
        color = next(colors)
        for t in thresholds:
            list_df = []
            for cls in cls_names:
                if generalized:
                    res_path = work_dir / (model_path + 'all') / f'identification_{cls}.csv'
                else:
                    res_path = work_dir / (model_path + cls) / f'identification.csv'

                df_ident = pd.read_csv(res_path)
                df_true = pd.read_csv(f'/mnt/data/afarec/data/PetFace/split/{cls}/identification_label.csv')
                df_true.rename(columns={'individual': 'test_label'}, inplace=True)
                df_merge = pd.merge(df_ident, df_true, on='test_label', how='left')
                list_df.append(df_merge)
            df_merge = pd.concat(list_df, axis='index')
            tpir, fpir, acc = calc_tpir_fpir_acc(t, df_merge)
            accs.append(acc*100)
            tpirs.append(tpir*100)
            fpirs.append(fpir*100)

        ax.plot(thresholds, tpirs, label=f'{model_name}-tpir', linestyle='--', color=color)
        ax.plot(thresholds, fpirs, label=f'{model_name}-fpir', linestyle='-', color=color)
        ax.plot(thresholds, accs, label=f'{model_name}-acc', linestyle=':', color=color)
        legend_handles.append(
            ax.plot([], [], color=color, marker='o', linestyle='None', markersize=14, label=model_name)[0]
        )

    # Add title and labels
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('TPIR / FPIR', fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    ax.tick_params(labelsize=20)
    ax.grid(True)

    legend_desc = ax.legend(handles=legend_handles_in, loc='upper right', fontsize=24)
    legend_names = ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1.05), fontsize=24)
    ax.add_artist(legend_desc)

    plt.tight_layout()
    if generalized:
        save_path = Path('/mnt/data/afarec/code/docs/figures/face_rec_tpir_fpir_gen.pdf')
    else:
        save_path = Path('/mnt/data/afarec/code/docs/figures/face_rec_tpir_fpir_spe.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # auc_table_verfication()
    # plot_dist()
    plot_acc_threshold(5, generalized=True)
    plot_acc_threshold(5, generalized=False)
    table_acc_identification()
    # plot_radar_topk()
