import os
import warnings

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import zscore
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.cluster import SpectralCoclustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
sns.set_theme()

plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=16)


def dimensionality_reduction(X, num_neighbors: int = 5, num_components: int = 2, min_dist: float = 0.1,
                             reduction_method: str = "umap", num_epochs: int = 2000, num_jobs: int = 2):
    num_examples, num_attributes = X.shape
    if reduction_method == "umap":
        init = "spectral"
        if num_attributes >= num_examples:
            init = "random"
        reducer = umap.UMAP(n_neighbors=num_neighbors, n_components=num_components, n_epochs=num_epochs, init=init,
                            min_dist=min_dist, n_jobs=num_jobs)
    elif reduction_method == "tsne":
        init = "pca"
        if num_attributes >= num_examples:
            init = "random"
        reducer = TSNE(n_components=num_components, perplexity=num_examples / 100, early_exaggeration=4,
                       learning_rate="auto", n_iter=num_epochs, init=init, random_state=12345, n_jobs=num_jobs)
    else:
        reducer = PCA(n_components=num_components, random_state=12345)
    X_reducer = reducer.fit_transform(X)
    return X_reducer


def clustering(X, cluster_type: str = "spectral", affinity: str = "nearest_neighbors", num_neighbors: int = 5,
               num_clusters: int = 4, num_jobs: int = 2, predict: bool = True):
    num_examples, num_attributes = X.shape
    if num_examples < num_clusters:
        num_clusters = num_examples
    if cluster_type == "kmeans":
        cls = MiniBatchKMeans(n_clusters=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "gmm":
        cls = GaussianMixture(n_components=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "hdbscan":
        cls = hdbscan.HDBSCAN(min_samples=num_neighbors, min_cluster_size=num_clusters)
    elif cluster_type == "spectral":
        if num_neighbors > num_examples:
            num_neighbors = num_examples
        cls = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack", n_neighbors=num_neighbors,
                                 affinity=affinity, n_init=100, assign_labels='discretize', n_jobs=num_jobs,
                                 random_state=12345)
    elif cluster_type == "cocluster":
        cls = SpectralCoclustering(n_clusters=num_clusters, svd_method="arpack", random_state=12345)
    elif cluster_type == "agglomerative":
        cls = AgglomerativeClustering(n_clusters=num_clusters, affinity='manhattan', linkage='single')
    elif cluster_type == "affinity":
        cls = AffinityPropagation(affinity='euclidean', random_state=12345)
    if predict:
        cls = cls.fit_predict(X)
    else:
        cls.fit(X)
    return cls


def plot_scatter(X, y, num_features: int = 100, legend_title: str = "Class", suptitle: str = "Hetero-Net",
                 file_name: str = "temp", save_path: str = "."):
    # plot figure
    plt.figure(figsize=(12, 8))
    sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette='tab10')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.suptitle('Using Top %s Features from %s' % (str(num_features), suptitle), fontsize=18, fontweight="bold")
    plt.legend(title=legend_title)
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


def plot_umap(X, y, num_features: int, standardize: bool = True, num_neighbors: int = 15, min_dist: float = 0,
              num_jobs: int = 2, suptitle: str = "Hetero-Net", file_name: str = "temp", save_path: str = "."):
    """
    UMAP results from Hetero-Net

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    min_dist : int (default=0)
        Minimum distance set for UMAP

    num_neighbors : int (default=15)
        Number of nearest neighbors for UMAP

    standardize : bool (default=True)
        Standardize data before UMAP

    Returns
    -------
    matplotlib plot of UMAP-ed data with top n_features from Hetero-Net
    """

    # standardize if needed
    if standardize:
        X = zscore(X)
    # make umap and umap data
    X_reducer = dimensionality_reduction(X, num_neighbors=num_neighbors, num_components=2, min_dist=min_dist,
                                         reduction_method="umap", num_epochs=2000, num_jobs=num_jobs)
    plot_scatter(X=X_reducer, y=y, num_features=num_features, suptitle=suptitle, file_name=file_name + "_umap.png",
                 save_path=save_path)


def plot_clusters(X, y, features_name: list, num_features: int, standardize: bool = True,
                  cluster_type: str = "spectral", num_clusters: int = 0, num_neighbors: int = 15, min_dist: float = 0,
                  heatmap: bool = False, proportion: bool = False, show_umap: bool = True, num_jobs: int = 2,
                  suptitle: str = "Hetero-Net", file_name: str = "temp", save_path: str = "."):
    """
    Cluster results from Hetero-Net

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    num_features : int
        Number of features to use for UMAP

    cluster_type : str (default=kmeans)
        Type of clustering algorithm to use for clustering data

    num_clusters : int (default=None)
        Pre-set number of clusters used, or leave as None to perform Silhoutte Analysis

    min_dist : int (default=0)
        Minimum distance set for UMAP

    num_neighbors : int (default=15)
        Number of nearest neighbors for UMAP

    standardize : bool (default=True)
        Standardize data before UMAP

    heatmap : bool (default=False)
        If true, will print heatmaps of clustered data with feature expression per cluster
    Returns
    -------
    matplotlib plot of UMAP-ed data with top n_features from Hetero-Net clustered
    """

    # standardize if needed
    if standardize:
        X = zscore(X)

    # check if dataframe, used to subset features later
    # if not isinstance(data, pd.DataFrame):
    #     raise Exception("Input data as pd.Dataframe with features as columns.")

    # make umap and umap data
    X_reducer = dimensionality_reduction(X, num_neighbors=num_neighbors, num_components=2, min_dist=min_dist,
                                         reduction_method="umap", num_epochs=2000, num_jobs=num_jobs)
    # Perform Silhoette Analysis
    silhouette_avg_n_clusters = []
    if num_clusters == 0:
        for i in range(2, 10):
            cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=i, num_jobs=num_jobs,
                                        predict=True)
            # find silhoette score
            silhouette_avg = silhouette_score(X_reducer, cluster_labels)
            silhouette_avg_n_clusters.append(silhouette_avg)
            # print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)
        # use highest silhoette score for clusters
        tmp = max(silhouette_avg_n_clusters)
        num_clusters = silhouette_avg_n_clusters.index(tmp) + 2
        cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=num_clusters,
                                    num_jobs=num_jobs, predict=True)
    else:
        cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=num_clusters,
                                    num_jobs=num_jobs, predict=True)
        tmp = 0

    if show_umap:
        plot_scatter(X=X_reducer, y=cluster_labels, num_features=num_features, legend_title="Cluster",
                     suptitle=file_name, file_name=file_name + "_umap_clusters.png", save_path=save_path)

    # print adjusted Rand score
    print("Rand Score " + str(rand_score(cluster_labels, y)))
    print("Adjusted Rand Score " + str(adjusted_rand_score(cluster_labels, y)))
    ars = adjusted_rand_score(cluster_labels, y)

    # print heatmap if true
    if heatmap:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = features_name
        X['cluster'] = cluster_labels
        df = X.groupby('cluster').mean()
        plot_heatmap(df, file_name=file_name + "_heatmap.png", save_path=save_path)

    # plot proportion bar chart
    if proportion and heatmap:
        dic = {}
        clus_labels = []
        for i in np.unique(cluster_labels):
            tmp_df = X[cluster_labels == i]
            issue = False
            issue_wk = []
            issue_index = []
            for cla in np.unique(y):
                if cla not in np.unique(tmp_df['class']):
                    tmp_df = pd.concat([tmp_df, pd.DataFrame({'class': cla}, index=[0])], ignore_index=True).fillna(0)
                    issue = True
                    issue_wk.append(cla)
            if issue == False:
                dic[i] = list(tmp_df['class'].value_counts().sort_index(ascending=False))
            else:
                tmp_ls = list(tmp_df['class'].value_counts().sort_index(ascending=False))
                for is_wk in issue_wk:
                    tmp_lss = list(np.unique(y))
                    tmp_lss.reverse()
                    issue_index.append(tmp_lss.index(is_wk))
                tmp_arr = np.array(tmp_ls)
                for issue in issue_index:
                    tmp_arr[issue] = 0
                dic[i] = list(tmp_arr)
            clus_labels.append("C" + str(i))

        # proportion of weeks
        df = pd.DataFrame(dic)[::-1]
        df.columns = clus_labels
        df.index = np.unique(y)

        df = df.T.div(df.sum(axis=0), axis=0)
        # plot
        plt.figure(figsize=(6, 5))
        df.plot(kind="bar", stacked=True)
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()

    return tmp, ars, num_clusters


def plot_boxplot(X, y, features_name, num_features: int, standardize: bool = False, swarmplot: bool = False):
    """
    Boxplots of top features from Hetero-Net

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    num_features : int
        Number of features to use for UMAP

    standardize : bool (default=True)
        Standardize data before UMAP

    swarmplot : bool (default=False)
        Determine if swarmplot is also used, helps see distribution better (computationally expensive with large dataset)

    Returns
    -------
    matplotlib boxplots of top features
    """
    # standardize if needed
    if standardize:
        X = zscore(X)

    # cycle through which features
    for i in range(0, num_features):
        # plot figure
        plt.figure(figsize=(6, 5))
        sns.boxplot(x=y, y=X[:, i], palette='tab10')
        if swarmplot:
            sns.swarmplot(x=y, y=X[:, i], color='black')
        plt.xlabel('Class')
        plt.ylabel("Expression values (Z-score)")
        plt.suptitle(features_name[i], fontsize=18, fontweight="bold")
        sns.despine()
