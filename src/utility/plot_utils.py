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
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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
    # plot figure
    plt.figure(figsize=(12, 8))
    sns.scatterplot(X_reducer[:, 0], X_reducer[:, 1], hue=y, palette='tab10')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.suptitle('Using Top %s Features from %s' % (str(num_features), suptitle), fontsize=18, fontweight="bold")
    plt.legend(title="Class")
    sns.despine()
    file_path = os.path.join(save_path, file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


##################################################################################################


# function to look at clustering results

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


def plot_clusters(X, y, heteronet_results, num_features: int, standardize: bool = True,
                  cluster_type: str = "spectral", num_clusters: int = None, num_neighbors: int = 15,
                  min_dist: float = 0, heatmap: bool = False, proportion: bool = False, plot: bool = True,
                  num_jobs: int = 2, suptitle: str = "Hetero-Net", file_name: str = "temp", save_path: str = "."):
    """
    Cluster results from Hetero-Net

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    heteronet_results : pd.DataFrame of results from Hetero-Net

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
    if num_clusters == None:
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
                                    num_jobs=num_jobs,
                                    predict=True)
        cluster_labels = cluster_labels + 1
        # print('Using ' + str(num_clusters) + ' to cluster data..')
        # print("Silhoette Score " + str(tmp))
    else:
        cluster_labels = clustering(X=X_reducer, cluster_type=cluster_type, num_clusters=num_clusters,
                                    num_jobs=num_jobs,
                                    predict=True)
        cluster_labels = cluster_labels + 1
        tmp = 0

    # print adjusted Rand score
    # print(df_umap['class'])
    print("Rand Score " + str(rand_score(cluster_labels, y)))
    print("Adjusted Rand Score " + str(adjusted_rand_score(cluster_labels, y)))
    ars = adjusted_rand_score(cluster_labels, y)

    if plot:
        # plot
        plt.figure(figsize=(6, 5))
        sns.scatterplot(X_reducer[:, 0], X_reducer[:, 1], hue=cluster_labels, palette='Dark2')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Using Top ' + str(num_features) + ' Features from Hetero-Net', fontsize=18)
        plt.legend(title="Cluster")
        sns.despine()

    # print heatmap if true
    if heatmap:
        ## intensity plot for deephetero
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = heteronet_results['features'][:num_features]
        X['cluster'] = cluster_labels

        data_gb = X.groupby('cluster').mean()

        plt.figure(figsize=(2.5, 3))
        # sns.heatmap(to_hm_colon.T,cbar_kws={'label': 'Normalized Average Expression'})
        cg = sns.clustermap(data_gb.T, col_cluster=False, cbar_pos=(.95, .08, .03, .7))
        ax = cg.ax_heatmap
        cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)
        cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)
        ax.set_xlabel('Cluster', fontsize=16)
        ax.set_ylabel('Top Features', fontsize=16)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        cg.ax_cbar.tick_params(labelsize=16)
        cg.ax_cbar.set_ylabel('Normalized Average Expression', fontsize=16)

    # plot proportion bar chart
    if proportion and heatmap:

        X['class'] = y

        dic = {}
        clus_labels = []
        for i in range(1, len(np.unique(X['cluster'])) + 1):
            tmp_df = X[X['cluster'] == i]
            issue = False
            issue_wk = []
            issue_index = []
            for cla in np.unique(X['class']):
                if cla not in np.unique(tmp_df['class']):
                    tmp_df = pd.concat([tmp_df, pd.DataFrame({'class': cla}, index=[0])], ignore_index=True).fillna(0)
                    issue = True
                    issue_wk.append(cla)
            if issue == False:
                dic[i] = list(tmp_df['class'].value_counts().sort_index(ascending=False))
            else:
                tmp_ls = list(tmp_df['class'].value_counts().sort_index(ascending=False))
                for is_wk in issue_wk:
                    tmp_lss = list(np.unique(X['class']))
                    tmp_lss.reverse()
                    issue_index.append(tmp_lss.index(is_wk))
                tmp_arr = np.array(tmp_ls)
                for issue in issue_index:
                    tmp_arr[issue] = 0
                dic[i] = list(tmp_arr)

            clus_labels.append("C" + str(i))

        # proportion of weeks
        df_prop = pd.DataFrame(dic)[::-1]
        df_prop.columns = clus_labels
        df_prop.index = np.unique(X['class'])

        df_prop = df_prop.T.div(df_prop.sum(axis=0), axis=0)

        # plot
        plt.figure(figsize=(6, 5))
        df_prop.plot(kind="bar", stacked=True)
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()
    return tmp, ars, num_clusters


##################################################################################################


# Function to look at boxplots of top features

def plot_boxplot(data, y, heteronet_results, n_features, standardize=False, swarmplot=False):
    """
    Boxplots of top features from Hetero-Net

    Parameters
    ----------
    data : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    heteronet_results : pd.DataFrame of results from Hetero-Net

    n_features : int
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
        data = zscore(data)

    # add class to dataframe
    data['class'] = y

    # cycle through which features
    for i in range(0, n_features):
        feat_int = heteronet_results['features'][i]
        # plot figure
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=data.loc[:, [feat_int, 'class']], x=data.loc[:, [feat_int, 'class']]['class'], y=feat_int,
                    palette='tab10')
        if swarmplot:
            sns.swarmplot(data=data.loc[:, [feat_int, 'class']], x=data.loc[:, [feat_int, 'class']]['class'],
                          y=feat_int, color='black')
        plt.xlabel('Class')
        sns.despine()


##################################################################################################


# SVM function to compare

##################################################################################################
