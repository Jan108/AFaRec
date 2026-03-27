import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = [
    "YuNet", "YuNet-s",
    "Retina R18", "Retina R34", "Retina MNet",
    "SCRFD 2.5G", "SCRFD 10G", "SCRFD 34G",
]

categories = [
    "Bird", "Cat", "Cat-Like", "Dog",
    "Dog-Like", "Horse-Like", "Small Animals", "All"
]

pretrained_scores = {
        "Retina R18": [53.51, 37.12, 41.80, 36.34, 43.27, 33.92, 45.92, 39.66],
        "Retina R34": [68.81, 54.30, 58.89, 46.86, 57.60, 41.62, 58.82, 52.14],
        "Retina MNet": [75.72, 56.85, 60.14, 52.17, 59.41, 42.47, 66.95, 56.34],
        "YuNet": [58.94, 37.27, 31.91, 29.40, 41.64, 33.50, 42.38, 37.10],
        "YuNet-s": [63.13, 40.74, 33.31, 32.16, 41.41, 36.94, 44.96, 39.88],
        "SCRFD 2.5G": [3.02, 2.40, 1.63, 2.62, 2.17, 1.76, 2.28, 2.13],
        "SCRFD 10G": [4.49, 3.96, 3.08, 4.03, 3.91, 2.94, 3.85, 3.58],
        "SCRFD 34G": [5.35, 4.13, 3.08, 4.54, 4.28, 3.46, 3.66, 3.83],
}
trained_scores = {
    "Retina R18": [43.36, 34.93, 31.59, 33.51, 34.20, 40.58, 35.00, 35.58],
    "Retina R34": [40.27, 40.94, 38.00, 39.64, 40.69, 42.08, 41.35, 39.89],
    "Retina MNet": [43.18, 38.63, 37.69, 36.49, 39.75, 40.61, 36.93, 38.39],
    "YuNet": [43.84, 51.16, 59.31, 45.48, 60.30, 41.41, 55.95, 49.18],
    "YuNet-s": [46.24, 44.32, 49.17, 36.92, 49.26, 40.39, 46.87, 43.34],
    "SCRFD 2.5G": [18.66, 29.79, 32.88, 27.03, 40.54, 23.82, 35.67, 27.50],
    "SCRFD 10G": [34.70, 58.12, 63.75, 47.95, 66.00, 52.58, 62.63, 52.16],
    "SCRFD 34G": [27.52, 53.96, 62.65, 40.75, 67.20, 42.68, 56.76, 45.36],
}
specialised_scores = {
    "Retina R18": [40.06, 30.15, 28.52, 31.86, 34.49, 35.57, 30.35, 32.01],
    "Retina R34": [44.37, 37.06, 35.62, 37.33, 40.00, 42.91, 38.49, 38.44],
    "Retina MNet": [41.49, 40.68, 36.75, 39.26, 39.56, 48.38, 41.36, 40.54],
    "YuNet": [47.30, 56.00, 70.87, 47.29, 69.55, 55.64, 62.78, 56.06],
    "YuNet-s": [40.68, 51.82, 56.57, 36.26, 59.62, 48.58, 55.92, 47.31],
    "SCRFD 2.5G": [23.85, 35.73, 39.37, 25.34, 34.57, 30.97, 37.94, 31.28],
    "SCRFD 10G": [34.38, 52.90, 57.73, 47.51, 46.76, 50.62, 61.42, 49.25],
    "SCRFD 34G": [25.88, 46.05, 50.99, 34.64, 40.64, 37.17, 55.50, 39.50],
}


def plot_figure(scores, name):
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
    ax.set_ylabel('Average Precision', fontsize=20)
    ax.set_title(f'{name} Model Performance nach Klasse', fontsize=20)
    ax.set_xticks(np.arange(n_categories) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(categories)
    ax.tick_params(labelsize=20)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=20)

    plt.tight_layout()
    plt.savefig(f'/mnt/data/afarec/code/docs/vortrag/model_performance_{name}.png', format='png', dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_figure(pretrained_scores, 'Pretrained')
    plot_figure(trained_scores, 'Generalisierende')
    plot_figure(specialised_scores, 'Specialisierende')
