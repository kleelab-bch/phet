import warnings
from typing import Literal

import numpy as np
import pandas as pd
import umap
from scipy.stats import zscore, gamma, scoreatpercentile, lognorm, expon
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.cluster import SpectralCoclustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score, f1_score, roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
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
        cls = HDBSCAN(min_cluster_size=5, min_samples=num_neighbors, max_cluster_size=None,
                      metric="euclidean", algorithm="auto", leaf_size=40, n_jobs=num_jobs,
                      cluster_selection_method="eom",
                      allow_single_cluster=False)
    elif cluster_type == "spectral":
        if num_neighbors > num_examples:
            num_neighbors = num_examples
        cls = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack", n_neighbors=num_neighbors,
                                 affinity=affinity, n_init=100, assign_labels='discretize', n_jobs=num_jobs,
                                 random_state=12345)
    elif cluster_type == "cocluster":
        cls = SpectralCoclustering(n_clusters=num_clusters, svd_method="arpack", random_state=12345)
    elif cluster_type == "agglomerative":
        cls = AgglomerativeClustering(n_clusters=num_clusters, metric=None)
    elif cluster_type == "affinity":
        cls = AffinityPropagation(affinity=affinity, random_state=12345)
    if predict:
        cls = cls.fit_predict(X)
    else:
        cls.fit(X)
    return cls


def significant_features(X, features_name, alpha: float = 0.05, X_map=None,
                         fit_type: Literal["expon", "gamma", "lognormal", "percentile"] = "gamma",
                         per: int = 95, map_genes: bool = True, ttest: bool = False):
    tempX = np.copy(X)
    if X.shape[1] != 1:
        tempX = X[:, 3]
    tempX += 0.001
    if fit_type == "percentile":
        temp = scoreatpercentile(tempX, per, interpolation_method="higher")
        selected_features = np.where(tempX > temp)[0]
    elif fit_type == "lognormal":
        shape, loc, scale = lognorm.fit(tempX)
        selected_features = np.where((1 - lognorm.cdf(tempX, shape, loc=loc, scale=scale)) < alpha)[0]
    elif fit_type == "expon":
        shape, scale = expon.fit(tempX)
        selected_features = np.where((1 - expon.cdf(tempX, shape, scale=scale)) < alpha)[0]
    else:
        shape, loc, scale = gamma.fit(zscore(tempX))
        selected_features = np.where((1 - gamma.cdf(zscore(tempX), shape, loc=loc, scale=scale)) < alpha)[0]
    if len(selected_features) != 0:
        X = X[selected_features]
        features_name = np.array(features_name)[selected_features].tolist()
    df = sort_features(X=X, features_name=features_name, X_map=X_map, map_genes=map_genes,
                       ttest=ttest)
    return df


def sort_features(X, features_name, X_map=None, map_genes: bool = True, ttest: bool = False,
                  ascending: bool = False):
    df = pd.concat([pd.DataFrame(features_name), pd.DataFrame(X)], axis=1)
    if X.shape[1] == 5:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=ascending).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=ascending).reset_index(drop=True)
    elif X.shape[1] == 6:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class', 'pvalue']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=ascending).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=ascending).reset_index(drop=True)
    elif X.shape[1] == 2:
        df.columns = ['features', 'score', 'pvalue']
        if ttest:
            df = df.sort_values(by=["pvalue"], ascending=ascending).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=ascending).reset_index(drop=True)
    elif X.shape[1] == 1:
        df.columns = ['features', 'score']
        df = df.sort_values(by=["score"], ascending=ascending).reset_index(drop=True)
    else:
        raise Exception("Please provide correct shape for X")
    if map_genes:
        df = df.merge(X_map, left_on='features', right_on='Probe Set ID')
        df = df.drop(["features"], axis=1)
    return df


def comparative_score(pred_features, true_features, metric: str = "f1"):
    if len(pred_features) != len(true_features):
        temp = "The number of samples must be same for both lists."
        raise Exception(temp)
    if metric == "f1":
        score = f1_score(y_true=true_features, y_pred=pred_features)
    elif metric == "precision":
        score = precision_score(y_true=true_features, y_pred=pred_features)
    elif metric == "recall":
        score = recall_score(y_true=true_features, y_pred=pred_features)
    elif metric == "auc":
        score = roc_auc_score(y_true=true_features, y_score=pred_features)
    elif metric == "accuracy":
        score = accuracy_score(y_true=true_features, y_pred=pred_features)
    else:
        score = jaccard_score(y_true=true_features, y_pred=pred_features)
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


def outliers_analysis(X, y, outliers: list, true_changed_features: list, pred_changed_features: list):
    # Detect outliers
    total_outliers = 0
    for group_idx in np.unique(y):
        examples_idx = np.where(y == group_idx)[0]
        q1 = np.percentile(X[examples_idx], q=25, axis=0)
        q3 = np.percentile(X[examples_idx], q=75, axis=0)
        iqr = q3 - q1  # Inter-quartile range
        fence_high = q3 + (1.5 * iqr)
        fence_low = q1 - (1.5 * iqr)
        for outlier_idx in outliers:
            for feature_idx in pred_changed_features:
                if feature_idx not in true_changed_features:
                    continue
                temp = np.where(X[examples_idx, feature_idx] > fence_high[feature_idx])[0]
                temp = examples_idx[temp]
                if outlier_idx in temp:
                    total_outliers += 1
                    continue
                temp = np.where(X[examples_idx, feature_idx] < fence_low[feature_idx])[0]
                temp = examples_idx[temp]
                if outlier_idx in temp:
                    total_outliers += 1
    return total_outliers


##############################################################
########### Intracluster and Intercluster metrics ############
##############################################################
# Intracluster -- Measuring distance between points in a cluster. 
# Lower values represetn densely packed clusters.
def complete_diameter_distance(X, y, metric: str = "euclidean",
                               num_jobs: int = 2):
    '''
    The farthest distance between two samples in a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            continue
        D = pairwise_distances(X=X[examples_idx], metric=metric, n_jobs=num_jobs)
        D = D / np.linalg.norm(D, axis=0)
        score += D.max()
    return score


def average_diameter_distance(X, y, metric: str = "euclidean",
                              num_jobs: int = 2):
    '''
    The average distance between all samples in a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            continue
        D = pairwise_distances(X=X[examples_idx], metric=metric, n_jobs=num_jobs)
        D = D / np.linalg.norm(D, axis=0)
        D = np.triu(D).sum()
        score += 1 / (len(examples_idx) * (len(examples_idx) - 1)) * D
    return score


def centroid_diameter_distance(X, y, metric: str = "euclidean",
                               num_jobs: int = 2):
    '''
    The double of average distance between samples and the center of a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            continue
        cluster_centers = X[examples_idx].sum(0) / len(examples_idx)
        cluster_centers = cluster_centers[None, :]
        D = pairwise_distances(X=X[examples_idx], Y=cluster_centers, metric=metric,
                               n_jobs=num_jobs)
        D = D / np.linalg.norm(D, axis=0)
        score += 2 * (D.sum() / len(examples_idx))
    return score


# Intercluster: measuring distance between clusters. 
# Higher values entail clusters are located farther apart.
def single_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The closest distance between two samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / np.linalg.norm(D, axis=0)
            score += D.min()
    return score


def maximum_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The farthest distance between two samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / np.linalg.norm(D, axis=0)
            score += D.max()
    return score


def average_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The average distance between all samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / np.linalg.norm(D, axis=0)
            score += D.sum() / (len(examples_i) * len(examples_j))
    return score


def centroid_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The distance between centers of clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = list()
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            center_i = X[examples_i].sum(0) / len(examples_i)
            center_i = center_i[None, :]
            center_j = X[examples_j].sum(0) / len(examples_j)
            center_j = center_j[None, :]
            D = pairwise_distances(X=center_i, Y=center_j, metric=metric,
                                   n_jobs=num_jobs)
            score.append(D.flatten()[0])
    score = np.array(score)
    score = score / np.linalg.norm(score)
    score = np.average(score)
    return score


def wards_distance(X, y):
    '''
    The different deviation between a group of 2 considered clusters and a
    "reputed" cluster joining those 2 clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = list()
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            center_i = X[examples_i].sum(0) / len(examples_i)
            center_j = X[examples_j].sum(0) / len(examples_j)
            temp = (2 * len(examples_i) * len(examples_j)) / (len(examples_i) + len(examples_j))
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            score.append(np.sqrt(temp * np.sum((center_i - center_j) ** 2)))
    score = np.array(score)
    score = score / np.linalg.norm(score)
    score = np.average(score)
    return score


def clustering_performance(X, labels_true, labels_pred, metric: str = "euclidean", num_jobs: int = 2):
    intra_complete = complete_diameter_distance(X=X, y=labels_pred, metric=metric,
                                                num_jobs=num_jobs)
    intra_average = average_diameter_distance(X=X, y=labels_pred, metric=metric,
                                              num_jobs=num_jobs)
    intra_centroid = centroid_diameter_distance(X=X, y=labels_pred, metric=metric,
                                                num_jobs=num_jobs)

    inter_single = single_linkage_distance(X=X, y=labels_pred, metric=metric,
                                           num_jobs=num_jobs)
    inter_maximum = maximum_linkage_distance(X=X, y=labels_pred, metric=metric,
                                             num_jobs=num_jobs)
    inter_average = average_linkage_distance(X=X, y=labels_pred, metric=metric,
                                             num_jobs=num_jobs)
    inter_centroid = centroid_linkage_distance(X=X, y=labels_pred, metric=metric,
                                               num_jobs=num_jobs)
    wards = wards_distance(X=X, y=labels_pred)
    silhouette = silhouette_score(X=X, labels=labels_pred, metric=metric)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true=labels_true,
                                                                              labels_pred=labels_pred)
    ari = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)
    ami = adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)

    list_scores = [intra_complete, intra_average, intra_centroid, inter_single,
                   inter_maximum, inter_average, inter_centroid, wards, silhouette,
                   homogeneity, completeness, v_measure, ari, ami]
    return list_scores


def clustering_performance_wo_ground_truth(X, labels_pred, metric: str = "euclidean", num_jobs: int = 2):
    intra_complete = complete_diameter_distance(X=X, y=labels_pred, metric=metric,
                                                num_jobs=num_jobs)
    intra_average = average_diameter_distance(X=X, y=labels_pred, metric=metric,
                                              num_jobs=num_jobs)
    intra_centroid = centroid_diameter_distance(X=X, y=labels_pred, metric=metric,
                                                num_jobs=num_jobs)

    inter_single = single_linkage_distance(X=X, y=labels_pred, metric=metric,
                                           num_jobs=num_jobs)
    inter_maximum = maximum_linkage_distance(X=X, y=labels_pred, metric=metric,
                                             num_jobs=num_jobs)
    inter_average = average_linkage_distance(X=X, y=labels_pred, metric=metric,
                                             num_jobs=num_jobs)
    inter_centroid = centroid_linkage_distance(X=X, y=labels_pred, metric=metric,
                                               num_jobs=num_jobs)
    wards = wards_distance(X=X, y=labels_pred)
    silhouette = silhouette_score(X=X, labels=labels_pred, metric=metric)
    calinski_harabasz = calinski_harabasz_score(X=X, labels=labels_pred)
    davies_bouldin = davies_bouldin_score(X=X, labels=labels_pred)

    list_scores = [intra_complete, intra_average, intra_centroid, inter_single,
                   inter_maximum, inter_average, inter_centroid, wards, silhouette,
                   calinski_harabasz, davies_bouldin]

    return list_scores
