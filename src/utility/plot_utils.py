import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import TomekLinks
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
from utility.utils import dimensionality_reduction, clustering, clustering_performance

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
sns.set_theme(style="white")

plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=16)


def make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


def plot_barplot(X, methods_name, metric: str = "f1", suptitle: str = "temp", file_name: str = "temp",
                 save_path: str = "."):
    plt.figure(figsize=(12, 8))
    plt.bar(x=methods_name, height=X)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Methods', fontsize=20, fontweight="bold")
    if metric == "f1":
        temp = "F1"
    elif metric == "precision":
        temp = "Precision"
    elif metric == "auc":
        temp = "AUC"
    elif metric == "accuracy":
        temp = "Accuracy"
    elif metric == "jaccard":
        temp = "Jaccard"
    elif metric == "ari":
        temp = "ARI"
    elif metric == "ami":
        temp = "AMI"
    else:
        raise Exception("Please provide a valid metric:f1, auc, jaccard, ari, and ami")
    plt.ylabel(temp + " scores of each method", fontsize=22)
    plt.title("Results using {0} data".format(suptitle), fontsize=26)
    file_path = os.path.join(save_path, file_name + "_" + temp.lower() + ".png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


def plot_scatter(X, y, num_features: int = 100, palette: dict = None, add_legend: bool = True,
                 legend_title: str = "Class", suptitle: str = "temp", file_name: str = "temp",
                 save_path: str = "."):
    plt.figure(figsize=(12, 10))
    plt.title('%s (%s features)' % (suptitle, str(num_features)), fontsize=36)
    if palette is None:
        palette = 'tab10'
    if y is not None:
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, s=80, alpha=0.6, linewidth=0,
                        legend=add_legend)
    else:
        add_legend = False
        sns.scatterplot(x=X[:, 0], y=X[:, 1], palette=palette, s=80, alpha=0.6, linewidth=0,
                        legend=add_legend)
    plt.xticks([], fontsize=28)
    plt.yticks([], fontsize=28)
    plt.xlabel("UMAP 1", fontsize=30)
    plt.ylabel("UMAP 2", fontsize=30)
    if add_legend:
        plt.legend(title=legend_title, fontsize=26, title_fontsize=30, markerscale=3)
    sns.despine()
    file_path = os.path.join(save_path, file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


def plot_heatmap(df, file_name: str = "temp", save_path: str = "."):
    plt.figure(figsize=(12, 14))
    cg = sns.clustermap(df.T, col_cluster=False, cbar_pos=(.95, .08, .03, .7))
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    cg.ax_cbar.tick_params(labelsize=16)
    cg.ax_cbar.set_ylabel('Normalized Average Expression', fontsize=16)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    ax = cg.ax_heatmap
    ax.set_xlabel('Cluster', fontsize=16)
    ax.set_ylabel('Top Features', fontsize=16)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    # plt.tight_layout()
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


def plot_umap(X, y, subtypes, features_name: list, num_features: int, labels_pred: list = [],
              perform_undersampling: bool = False, standardize: bool = True,
              num_neighbors: int = 15, min_dist: float = 0, perform_cluster: bool = False,
              cluster_type: str = "spectral", num_clusters: int = 0,
              max_clusters: int = 10, heatmap_plot: bool = True, palette: dict = None,
              num_jobs: int = 2, suptitle: str = "temp", file_name: str = "temp",
              save_path: str = "."):
    score = 0
    # Standardize if needed
    if standardize:
        X = zscore(X)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Make umap and umap data
    X_reducer = dimensionality_reduction(X, num_neighbors=num_neighbors, num_components=2, min_dist=min_dist,
                                         reduction_method="umap", num_epochs=2000, num_jobs=num_jobs)
    df = pd.DataFrame(X_reducer, columns=["UMAP1", "UMAP2"])
    df.to_csv(os.path.join(save_path, file_name + "_umap.csv"), sep=',')

    # Perform under-sampling
    if perform_undersampling:
        list_downsamples = list()
        temp = int(np.mean(list(Counter(subtypes).values())))
        # temp = 500
        for subtype in set(subtypes):
            temp_idx = [idx for idx, t in enumerate(subtypes) if t == subtype]
            if len(temp_idx) >= temp:
                temp_idx = resample(temp_idx, replace=False, n_samples=temp, random_state=12345)
            list_downsamples.extend(temp_idx)
        X_reducer = X_reducer[list_downsamples]
        X = X[list_downsamples]
        y = y[list_downsamples]
        subtypes = list(np.array(subtypes)[list_downsamples])

        # With TomekLinks cleaning method, the number of samples in each class will not be equalized even if targeted.
        us = TomekLinks(sampling_strategy="not minority", n_jobs=num_jobs)
        le = LabelEncoder()
        subtypes = le.fit_transform(subtypes)
        X_reducer, subtypes = us.fit_resample(X=X_reducer, y=subtypes)
        X = X[us.sample_indices_]
        y = y[us.sample_indices_]
        subtypes = list(le.inverse_transform(subtypes))
        # print(y.shape)

    plot_scatter(X=X_reducer, y=None, num_features=num_features, palette=None, legend_title="Class",
                 suptitle=suptitle, file_name=file_name + "_no_class_umap.png", save_path=save_path)
    plot_scatter(X=X_reducer, y=y, num_features=num_features, palette=None, legend_title="Class",
                 suptitle=suptitle, file_name=file_name + "_umap.png", save_path=save_path)
    if subtypes is not None:
        plot_scatter(X=X_reducer, y=subtypes, num_features=num_features, palette=palette, suptitle=suptitle,
                     file_name=file_name + "_subtypes_umap.png", save_path=save_path)
        plot_scatter(X=X_reducer, y=subtypes, num_features=num_features, palette=palette, add_legend=False,
                     suptitle=suptitle, file_name=file_name + "_subtypes_nolegend_umap.png",
                     save_path=save_path)

    if perform_cluster:
        labels_true = np.unique(subtypes)
        labels_true = dict([(item, idx) for idx, item in enumerate(labels_true)])
        labels_true = [labels_true[item] for item in subtypes]
        scores_list = []
        if num_clusters == 0:
            for i in range(2, max_clusters):
                labels_pred = clustering(X=X, cluster_type=cluster_type, num_clusters=i,
                                         num_jobs=num_jobs, predict=True)
                score = silhouette_score(X, labels_pred)
                scores_list.append(score)
            # use highest score for clusters
            temp = max(scores_list)
            num_clusters = scores_list.index(temp) + 2
            labels_pred = clustering(X=X, cluster_type=cluster_type, num_clusters=num_clusters,
                                     num_jobs=num_jobs, predict=True)
        else:
            if len(labels_pred) == 0:
                labels_pred = clustering(X=X, cluster_type=cluster_type, num_clusters=num_clusters,
                                         num_jobs=num_jobs, predict=True)
        df = pd.DataFrame(labels_pred, columns=["Cluster"])
        df.to_csv(os.path.join(save_path, file_name + "_clusters.csv"), sep=',')

        # Plot scatter 
        plot_scatter(X=X_reducer, y=labels_pred, num_features=num_features, palette=None, legend_title="Cluster",
                     suptitle=suptitle, file_name=file_name + "_clusters_umap.png", save_path=save_path)

        if subtypes is not None:
            labels_true = np.array(labels_true)
            labels_pred = np.array(labels_pred)
            list_scores = clustering_performance(X=X, labels_true=labels_true, labels_pred=labels_pred,
                                                 num_jobs=num_jobs)
            temp = np.unique(labels_pred).shape[0]
            if temp > np.unique(labels_true).shape[0]:
                temp = np.unique(labels_true).shape[0] / temp
            else:
                temp /= np.unique(labels_true).shape[0]
            list_scores = [score * temp for score in list_scores]

    if heatmap_plot and perform_cluster:
        # print heatmap if true
        scaler = MinMaxScaler()
        scaler.fit(X)
        df = pd.DataFrame(scaler.transform(X))
        df.columns = features_name
        df['cluster'] = labels_pred
        df = df.groupby('cluster').mean()
        if df.shape[1] > 2:
            plot_heatmap(df, file_name=file_name + "_heatmap.png", save_path=save_path)

    if perform_cluster and subtypes is not None:
        return list_scores


def plot_boxplot(X, y, features_name, num_features: int, standardize: bool = False, swarmplot: bool = False):
    # standardize if needed
    if standardize:
        X = zscore(X)

    # cycle through which features
    for feature_idx in range(num_features):
        # plot figure
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=y, y=X[:, feature_idx], palette='tab10')
        if swarmplot:
            sns.swarmplot(x=y, y=X[:, feature_idx], color='black')
        plt.xlabel('Class')
        plt.ylabel("Expression values (Z-score)")
        plt.suptitle(features_name[feature_idx], fontsize=18, fontweight="bold")
        sns.despine()
