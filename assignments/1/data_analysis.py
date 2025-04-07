import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_scatter_bw_features(feature1:str, feature2:str) -> None:
    reqs = [feature1, feature2,'track_genre']
    df = pd.read_csv('../../data/external/spotify.csv')
    df = df.drop_duplicates(subset='track_id', keep='first')
    df = df[reqs]

    x = df[feature1]
    y = df[feature2]

    grouped = df.groupby('track_genre')
    colors = plt.get_cmap('Spectral', len(grouped)) 
    fig, ax = plt.subplots(figsize=(16,9))
    idx  = 0
    for name, group in grouped:
        ax.scatter(group[feature1], group[feature2], label=name, color=colors(idx / len(grouped)))
        idx += 1

    ax.set_title(f'Scatter Plot : {feature1} vs. {feature2} by Genre')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    ax.legend(title='Track Genre')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=4, markerscale=1, frameon=True)
    plt.subplots_adjust(right=0.6)
    plt.savefig(f'figures/scatt_{feature1}_vs_{feature2}.png')
    # plt.show()
    plt.close()

def plot_histogram(features:list):
    n_cols = 4
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    fig.suptitle('Histograms of Features', fontsize=16)
    axes = axes.flatten()
    data = data[:, :-1] 
    data = data.astype(np.float64)

    for idx, feature in enumerate(features):
        axs = axes[idx]

        dataf = data[:,idx]
        axs.hist(dataf, bins=30, color='skyblue', edgecolor='black')
        axs.set_title(feature)
        axs.set_xlabel('Value')
        axs.set_ylabel('Frequency')

    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig('figures/feature_histogram.png')
    # plt.show()
    plt.close()


def plot_boxplot(feature:str) -> None:
    reqs = [feature, 'track_genre']
    df = pd.read_csv('../../data/external/spotify.csv')
    df = df.drop_duplicates(subset='track_id', keep='first')
    df = df[reqs]

    unique_classes = df['track_genre'].unique()
    slot_size = 10  
    num_plots = (len(unique_classes) + slot_size - 1) // slot_size

    y_min = df[feature].min()
    y_max = df[feature].max()
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    plt.suptitle(f'{feature} Box Plot')
    axes = axes.flatten()  

    for i in range(num_plots):
        classes_to_plot = unique_classes[i * slot_size:(i + 1) * slot_size]
        df_subset = df[df['track_genre'].isin(classes_to_plot)]
        sns.boxplot(data=df_subset, y=feature, x='track_genre', ax=axes[i])
        axes[i].set_ylim(y_min, y_max)
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(45)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'figures/{feature}_box_plot.png')
    # plt.show()
    plt.close()



