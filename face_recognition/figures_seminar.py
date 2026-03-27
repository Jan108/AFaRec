import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = [
    "PetFace",
    "GhostV2-Arc", "GhostV2-Cos",
    "ArcFace-R34", "ArcFace-R50",
    "CosFace-R34", "CosFace-R50",
    "SphereFace20", "SphereFace64",
]

veri_categories = [
    "Bird", "Cat", "Dog", "Small Animals", "All"
]
ident_categories = [
    "Bird", "Cat", "Dog", "Small Animals", "All", 'TPIR', 'FPIR'
]


veri_generalized_scores = {
    "PetFace": [93.89, 98.04, 99.01, 91.12, 95.52],
    "GhostV2-Arc": [94.61, 98.10, 98.73, 94.92, 98.16],
    "GhostV2-Cos": [94.96, 97.94, 98.58, 94.61, 98.00],
    "ArcFace-R34": [92.72, 97.55, 98.14, 92.26, 97.54],
    "ArcFace-R50": [93.05, 97.53, 98.29, 92.71, 97.60],
    "CosFace-R34": [91.81, 97.60, 98.16, 92.94, 97.59],
    "CosFace-R50": [92.93, 97.51, 98.14, 92.43, 97.52],
    "SphereFace20": [84.44, 95.15, 95.14, 84.19, 94.65],
    "SphereFace64": [81.94, 95.10, 94.83, 83.28, 94.51]
}
veri_specialized_scores = {
    'PetFace': [0,0,0,0, 91.30],
    "GhostV2-Arc": [87.88, 97.27, 98.09, 88.09, 97.14],
    "GhostV2-Cos": [87.05, 97.89, 97.05, 86.93, 97.15],
    "ArcFace-R34": [80.42, 97.02, 93.70, 77.68, 95.12],
    "ArcFace-R50": [81.64, 97.12, 93.75, 76.25, 95.11],
    "CosFace-R34": [78.61, 96.95, 92.81, 78.57, 94.80],
    "CosFace-R50": [80.24, 96.85, 94.08, 76.62, 95.06],
    "SphereFace20": [65.57, 95.24, 93.21, 59.80, 92.52],
    "SphereFace64": [50.00, 95.07, 93.10, 72.54, 92.60]
}


ident_generalized_scores = {
    "GhostV2-Arc": [64.50, 87.00, 82.50, 69.75, 75.94, 56.62, 4.75],
    "GhostV2-Cos": [59.00, 82.00, 77.50, 64.50, 70.75, 43.88, 2.38],
    "ArcFace-R34": [52.00, 82.50, 75.00, 58.75, 67.06, 35.88, 1.75],
    "ArcFace-R50": [53.50, 83.75, 76.25, 60.75, 68.56, 38.88, 1.75],
    "CosFace-R34": [53.25, 81.25, 74.75, 59.00, 67.06, 36.62, 2.50],
    "CosFace-R50": [54.00, 83.00, 76.25, 60.00, 68.31, 39.62, 3.00],
    "SphereFace20": [51.00, 71.25, 64.25, 52.75, 59.81, 23.00, 3.38],
    "SphereFace64": [49.00, 70.75, 63.25, 52.25, 58.81, 20.50, 2.88]
}
ident_specialized_scores = {
    "GhostV2-Arc": [52.75, 85.75, 79.75, 57.50, 68.94, 41.50, 3.62],
    "GhostV2-Cos": [51.00, 81.25, 67.75, 54.50, 63.62, 29.88, 2.62],
    "ArcFace-R34": [47.75, 86.75, 66.00, 49.50, 62.50, 29.50, 4.50],
    "ArcFace-R50": [47.50, 88.25, 65.50, 51.50, 63.19, 30.00, 3.62],
    "CosFace-R34": [49.50, 86.00, 64.75, 51.00, 62.81, 29.12, 3.50],
    "CosFace-R50": [49.50, 86.75, 68.25, 51.00, 63.88, 31.13, 3.38],
    "SphereFace20": [49.75, 82.75, 66.25, 50.00, 62.19, 26.75, 2.38],
    "SphereFace64": [0.25, 82.50, 65.50, 50.50, 49.69, 27.12, 27.75]
}


def plot_figure(scores, name, categories, ylabel):
    # Matplotlib default colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(18, 8))

    # Number of categories and models
    n_categories = len(categories)
    n_models = len(models)

    # Bar width
    bar_width = 0.1

    # Plot bars for each model
    for i, model in enumerate(models):
        ax.bar(
            np.arange(n_categories) + i * bar_width,
            scores[model],
            width=bar_width,
            label=model,
            color=colors[i]
        )

    # Customize the plot
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(f'{name} Model Performance nach Klasse', fontsize=20)
    ax.set_xticks(np.arange(n_categories) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(categories)
    ax.tick_params(labelsize=20)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=20)

    plt.tight_layout()
    plt.savefig(f'/mnt/data/afarec/code/docs/vortrag/face_{ylabel}_{name}.png', format='png', dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_figure(veri_generalized_scores, 'Generalisierende', veri_categories, 'Area under Curve')
    plot_figure(veri_specialized_scores, 'Specialisierende', veri_categories, 'Area under Curve')
    # plot_figure(ident_generalized_scores, 'Generalisierende', ident_categories, 'Accuracy')
    # plot_figure(ident_specialized_scores, 'Spezialisierende', ident_categories, 'Accuracy')
