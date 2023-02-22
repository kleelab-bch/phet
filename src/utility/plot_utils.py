import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyCompare
import seaborn as sns
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from utility.utils import dimensionality_reduction, clustering

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
    elif metric == "auc":
        temp = "AUC"
    elif metric == "accuracy":
        temp = "Accuracy"
    elif metric == "jaccard":
        temp = "Jaccard"
    elif metric == "ari":
        temp = "ARI"
    else:
        raise Exception("Please provide a valid metric:f1, auc, jaccard, and ari")
    plt.ylabel(temp + " scores of each method", fontsize=22)
    plt.title("Results using {0} data".format(suptitle), fontsize=26)
    file_path = os.path.join(save_path, file_name + "_" + temp.lower() + ".png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


def plot_scatter(X, y, num_features: int = 100, add_legend:bool=True, legend_title: str = "Class", 
                 suptitle: str = "temp", file_name: str = "temp", save_path: str = "."):
    plt.figure(figsize=(12, 10))
    plt.title('%s (%s features)' % (suptitle, str(num_features)), fontsize=36)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='tab10', s=80, alpha=0.6, linewidth=0, 
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


def plot_umap(X, y, subtypes, features_name: list, num_features: int, standardize: bool = True, num_neighbors: int = 15,
              min_dist: float = 0, perform_cluster: bool = False, cluster_type: str = "spectral", num_clusters: int = 0,
              max_clusters: int = 10, apply_hungarian: bool = False, heatmap_plot: bool = True, num_jobs: int = 2,
              suptitle: str = "temp", file_name: str = "temp", save_path: str = "."):
    score = 0
    # Standardize if needed
    if standardize:
        X = zscore(X)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Make umap and umap data
    X_reducer = dimensionality_reduction(X, num_neighbors=num_neighbors, num_components=2, min_dist=min_dist,
                                         reduction_method="umap", num_epochs=2000, num_jobs=num_jobs)
    plot_scatter(X=X_reducer, y=y, num_features=num_features, legend_title="Class",
                 suptitle=suptitle, file_name=file_name + "_umap.png", save_path=save_path)
    if subtypes is not None:
        plot_scatter(X=X_reducer, y=subtypes, num_features=num_features, suptitle=suptitle,
                     file_name=file_name + "_subtypes_umap.png", save_path=save_path)
        plot_scatter(X=X_reducer, y=subtypes, num_features=num_features, add_legend=False, 
                     suptitle=suptitle, file_name=file_name + "_subtypes_nolegend_umap.png", 
                     save_path=save_path)

    if perform_cluster:
        # Perform Silhoette Analysis
        silhouette_avg_n_clusters = []
        if num_clusters == 0:
            for i in range(2, max_clusters):
                cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=i, num_jobs=num_jobs,
                                            predict=True)
                silhouette_avg = silhouette_score(X_reducer, cluster_labels)
                silhouette_avg_n_clusters.append(silhouette_avg)
            # use highest silhoette score for clusters
            temp = max(silhouette_avg_n_clusters)
            num_clusters = silhouette_avg_n_clusters.index(temp) + 2
            cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=num_clusters,
                                        num_jobs=num_jobs, predict=True)
        else:
            cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=num_clusters,
                                        num_jobs=num_jobs, predict=True)
        df = pd.DataFrame(cluster_labels, columns=["Cluster"])
        df.to_csv(os.path.join(save_path, file_name + "_clusters.csv"), sep=',')

        # Plot scatter 
        plot_scatter(X=X_reducer, y=cluster_labels, num_features=num_features, legend_title="Cluster",
                     suptitle=suptitle, file_name=file_name + "_clusters_umap.png", save_path=save_path)

        if subtypes is not None:
            y_true = np.unique(subtypes)
            y_true = dict([(item, idx) for idx, item in enumerate(y_true)])
            y_true = [y_true[item] for item in subtypes]
            if apply_hungarian:
                # Match cluster labels to true labels
                cm = confusion_matrix(y_true=y_true, y_pred=cluster_labels)
                # Permutation maximizing the sum of the diagonal elements using the Hungarian algorithm
                indexes = linear_assignment(make_cost_m(cm))
                js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
                cm = cm[:, js]
                score = np.trace(cm) / np.sum(cm)
            else:
                score = adjusted_rand_score(labels_true=y_true, labels_pred=cluster_labels)

    if heatmap_plot and perform_cluster:
        # print heatmap if true
        scaler = MinMaxScaler()
        scaler.fit(X)
        df = pd.DataFrame(scaler.transform(X))
        df.columns = features_name
        df['cluster'] = cluster_labels
        df = df.groupby('cluster').mean()
        if df.shape[1] > 2:
            plot_heatmap(df, file_name=file_name + "_heatmap.png", save_path=save_path)

    if perform_cluster and subtypes is not None:
        return score


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


def plot_blandaltman(X, y, features_name, num_features: int, standardize: bool = False, save_path: str = "."):
    if np.unique(y).shape[0] != 2:
        raise Exception("Bland–Altman plot supports only two groups!")
    # standardize if needed
    if standardize:
        X = zscore(X)

    classes = np.unique(y)
    # cycle through which features
    for feature_idx in range(num_features):
        examples1 = np.where(y == classes[0])[0]
        examples2 = np.where(y == classes[1])[0]
        file_path = os.path.join(save_path, features_name[feature_idx] + "_blandAltman.png")
        pyCompare.blandAltman(X[examples1, feature_idx], X[examples2, feature_idx], savePath=file_path)
