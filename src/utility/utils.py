import warnings

import hdbscan
import numpy as np
import pandas as pd
import umap
from scipy.stats import zscore, gamma
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.cluster import SpectralCoclustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from sklearn.mixture import GaussianMixture

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


def dimensionality_reduction(X, num_neighbors: int = 5, num_components: int = 2, min_dist: float = 0.1,
                             reduction_method: str = "umap", num_epochs: int = 2000, num_jobs: int = 2):
    num_examples, num_features = X.shape
    if reduction_method == "umap":
        init = "spectral"
        if num_features >= num_examples:
            init = "random"
        reducer = umap.UMAP(n_neighbors=num_neighbors, n_components=num_components, n_epochs=num_epochs, init=init,
                            min_dist=min_dist, n_jobs=num_jobs)
    elif reduction_method == "tsne":
        init = "pca"
        if num_features >= num_examples:
            init = "random"
        reducer = TSNE(n_components=num_components, perplexity=num_examples / 100, early_exaggeration=4,
                       learning_rate="auto", n_iter=num_epochs, init=init, random_state=12345, n_jobs=num_jobs)
    else:
        reducer = PCA(n_components=num_components, random_state=12345)
    X_reducer = reducer.fit_transform(X)
    return X_reducer


def clustering(X, cluster_type: str = "spectral", affinity: str = "nearest_neighbors", num_neighbors: int = 5,
               num_clusters: int = 4, num_jobs: int = 2, predict: bool = True):
    num_examples, num_features = X.shape
    if num_examples < num_clusters:
        num_clusters = num_examples
    if cluster_type == "kmeans":
        cls = MiniBatchKMeans(n_clusters=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "gmm":
        cls = GaussianMixture(n_components=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "hdbscan":
        cls = hdbscan.HDBSCAN(min_samples=num_neighbors, min_cluster_size=5, allow_single_cluster=False,
                              core_dist_n_jobs=num_jobs)
    elif cluster_type == "spectral":
        if num_neighbors > num_examples:
            num_neighbors = num_examples
        cls = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack", n_neighbors=num_neighbors,
                                 affinity=affinity, n_init=100, assign_labels='discretize', n_jobs=num_jobs,
                                 random_state=12345)
    elif cluster_type == "cocluster":
        cls = SpectralCoclustering(n_clusters=num_clusters, svd_method="arpack", random_state=12345)
    elif cluster_type == "agglomerative":
        cls = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity)
    elif cluster_type == "affinity":
        cls = AffinityPropagation(affinity=affinity, random_state=12345)
    if predict:
        cls = cls.fit_predict(X)
    else:
        cls.fit(X)
    return cls


def significant_features(X, features_name, pvalue: float = 0.05, X_map=None, map_genes: bool = True,
                         ttest: bool = False):
    tempX = np.copy(X)
    if X.shape[1] != 1:
        tempX = X[:, 3]
    shape, loc, scale = gamma.fit(zscore(tempX))
    selected_features = np.where((1 - gamma.cdf(zscore(tempX), shape, loc=loc, scale=scale)) <= pvalue)[0]
    if len(selected_features) != 0:
        X = X[selected_features]
        features_name = np.array(features_name)[selected_features].tolist()
    df = sort_features(X=X, features_name=features_name, X_map=X_map, map_genes=map_genes,
                       ttest=ttest)
    return df


def sort_features(X, features_name, X_map=None, map_genes: bool = True, ttest: bool = False):
    df = pd.concat([pd.DataFrame(features_name), pd.DataFrame(X)], axis=1)
    if X.shape[1] == 5:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    elif X.shape[1] == 6:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class', 'pvalue']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    elif X.shape[1] == 2:
        df.columns = ['features', 'score', 'pvalue']
        if ttest:
            df = df.sort_values(by=["pvalue"], ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    elif X.shape[1] == 1:
        df.columns = ['features', 'score']
        df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    else:
        raise Exception("Please provide correct shape for X")
    if map_genes:
        df = df.merge(X_map, left_on='features', right_on='Probe Set ID')
        df = df.drop(["features"], axis=1)
    return df


def comparative_score(top_features_pred, top_features_true):
    if len(top_features_pred) != len(top_features_true):
        temp = "The number of samples must be same for both lists."
        raise Exception(temp)
    score = jaccard_score(y_true=top_features_true, y_pred=top_features_pred)
    return score


def outliers_analysis(X, y, regulated_features: list):
    # Get feature size
    num_features = X.shape[1]

    regulated_features = np.where(regulated_features != 0)[0]

    # Detect outliers
    outliers_dict = dict()
    for group_idx in np.unique(y):
        examples_idx = np.where(y == group_idx)[0]
        q1 = np.percentile(X[examples_idx], q=25, axis=0)
        q3 = np.percentile(X[examples_idx], q=75, axis=0)
        iqr = q3 - q1  # Inter-quartile range
        fence_high = q3 + (1.5 * iqr)
        fence_low = q1 - (1.5 * iqr)
        temp = list()
        for feature_idx in range(num_features):
            temp1 = np.where(X[examples_idx, feature_idx] > fence_high[feature_idx])[0]
            temp2 = np.where(X[examples_idx, feature_idx] < fence_low[feature_idx])[0]
            temp.append(temp1.tolist() + temp2.tolist())
        outliers_dict.update({group_idx: temp})
    del temp, temp1, temp2

    # Calculate the outliers number and properties
    for group_idx, group_item in outliers_dict.items():
        num_outliers = 0
        num_regulated_outliers = 0
        for feature_idx, sample_list in enumerate(group_item):
            if len(sample_list) > 0:
                num_outliers += len(sample_list)
                if feature_idx in regulated_features:
                    num_regulated_outliers += len(sample_list)
                    # print(">> Feature: {0}; Group: {1}; Outliers: {2}".format(feature_idx, group_idx, sample_list))
        print("\t>> Group: {0}; Expected outliers per feature: {1:.4f}".format(group_idx, num_outliers / num_features))
        print("\t>> Group: {0}; Expected outliers per expressed feature: {1:.4f}".format(group_idx,
                                                                                         num_regulated_outliers / len(
                                                                                             regulated_features)))
