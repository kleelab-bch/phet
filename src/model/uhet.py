import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import iqr
from scipy.stats import zscore

from utility.file_path import DATASET_PATH


def heteroiqr(X, y):
    """
    Hetero-Net Function

    Perform Deep Metric Learning with UMAP-based clustering to find subpopulations of classes

    Read more in the USER GUIDE

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples

    standardize : bool, default=True
        Standardizes data using zscore (NOTE: test more?)

    Attributes
    ----------
    features_ : list
        The features found by the algorithm. Ranked in order of importance

    score_ : list
        The score found by the algorithm for each feature


    References
    ----------
    NOTE: MINE


    Examples
    ----------
    NOTE: TODO
    """

    # Extract properties
    num_classes = len(np.unique(y))
    num_features = X.shape[1]

    # Robustly estimate median by classes
    med = list()
    for i in range(num_classes):
        example_idx = np.where(y == i)[0]
        example_med = np.median(X[example_idx], axis=0)
        temp = np.absolute(X[example_idx] - example_med)
        med.append(temp)
    med = np.median(np.concatenate(med), axis=0)
    X = X / med

    # make transposed matrix with shape (feat per class, observation per class)
    # find mean and iqr difference between genes
    var_ls = []
    mean_ls = []
    ttest_ls = []
    what_class = []
    for p in range(num_features):
        temp_lst = []
        temp_mean_lst = []
        temp_ttest_lst = []
        for i in range(num_classes):
            examples_i = np.where(y == i)[0]
            for j in range(i + 1, num_classes):
                examples_j = np.where(y == j)[0]
                temp = iqr(X[examples_i, p], rng=(25, 75), scale=1.0)
                temp = temp - iqr(X[examples_j, p], rng=(25, 75), scale=1.0)
                temp_lst.append(temp)
                temp = np.mean(X[examples_i, p])
                temp = temp - np.mean(X[examples_j, p])
                temp_mean_lst.append(temp)
                temp = stats.ttest_ind(X[examples_i, p], X[examples_j, p])[0]
                temp_ttest_lst.append(temp)

        # check if negative to seperate classes for later
        if max(temp_lst) <= 0:
            what_class.append(0)
        else:
            what_class.append(1)

        # append the top variance
        var_ls.append(max(np.abs(temp_lst)))
        mean_ls.append(max(np.abs(temp_mean_lst)))
        ttest_ls.append(max(np.abs(temp_ttest_lst)))

    results = pd.concat([pd.DataFrame(var_ls)], axis=1)
    results.columns = ['iqr']
    results['median_diff'] = mean_ls
    results['ttest'] = ttest_ls
    results['score'] = np.array(mean_ls) + np.array(var_ls)
    results['class_diff'] = what_class

    return results.to_numpy()


def copa_statistic(X, y, q: float = 0.75, test_class: int = 1):
    # Compute column-wise the median of expression values
    # and the median absolute deviation of expression values 
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.absolute(X - med), axis=0)

    # Include only test data 
    X = X[np.where(y == test_class)[0]]

    # Calculate statistics
    results = (np.percentile(a=X, q=100 * q, axis=0) - med) / mad

    return results


def outlier_sum_statistic(X, y, q: float = 0.75, test_class: int = 1, two_sided_test: bool = True):
    num_features = X.shape[1]

    # Robustly standarize median
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.absolute(X - med), axis=0)
    X = (X - med) / mad

    # IQR estimation
    interquartile_range = iqr(X, axis=0, rng=(25, 75), scale=1.0)
    qr_pos = np.percentile(a=X, q=q, axis=0)
    qriqr_pos = qr_pos + interquartile_range
    qr_neg = np.percentile(a=X, q=1 - q, axis=0)
    qriqr_neg = qr_neg - interquartile_range

    # Include only test data 
    X = X[np.where(y == test_class)[0]]

    # Find one sided or two-sided stat
    os_pos = [X[np.where(X[:, idx] > qriqr_pos[idx])[0], idx].sum()
              for idx in range(num_features)]
    if two_sided_test:
        os_neg = [X[np.where(X[:, idx] < qriqr_neg[idx])[0], idx].sum()
                  for idx in range(num_features)]
    else:
        os_neg = np.zeros_like(os_pos)
    results = np.max(np.c_[np.absolute(os_pos), np.absolute(os_neg)], axis=1)

    return results


def outlier_robust_tstatistic(X, y, q: float = 0.75, normal_class: int = 0, test_class: int = 1):
    num_features = X.shape[1]

    # Robustly estimate median by classes
    normal_idx = np.where(y == normal_class)[0]
    med_normal = np.median(X[normal_idx], axis=0)
    test_idx = np.where(y == test_class)[0]
    med_test = np.median(X[test_idx], axis=0)
    X_normal = np.absolute(X[normal_idx] - med_normal)
    X_test = np.absolute(X[test_idx] - med_test)
    med = np.concatenate((X_normal, X_test))
    med = np.median(med, axis=0)

    # IQR estimation
    interquartile_range = iqr(X[normal_idx], axis=0, rng=(25, 75), scale=1.0)
    qr = np.percentile(a=X[normal_idx], q=q, axis=0)
    qriqr = qr + interquartile_range

    # Get samples indices
    u = [np.where(X[test_idx, feature_idx] > qriqr[feature_idx])[0]
         for feature_idx in range(num_features)]
    X = X[test_idx]

    # Compute ORT test
    results = list()
    for feature_idx in range(num_features):
        temp = np.sum(X[u[feature_idx], feature_idx] - med_normal[feature_idx])
        temp = temp / med[feature_idx]
        results.append(temp)

    return np.array(results)


def maximum_ordered_subset_tstatistics(X, y, q: float = 0.75, normal_class: int = 0, test_class: int = 1):
    pass


def least_sum_ordered_subset_square_tstatistic(X, y, q: float = 0.75, normal_class: int = 0, test_class: int = 1):
    pass


def dids(X, y, q: float = 0.75, normal_class: int = 0, test_class: int = 1):
    pass


def top_features(X, features_name, map_genes: bool = True, ttest: bool = False):
    df = pd.concat([pd.DataFrame(features_name), pd.DataFrame(X)], axis=1)
    if len(X.shape) > 1:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    else:
        df.columns = ['features', 'score']
        df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    if map_genes:
        df = df.merge(hu6800, left_on='features', right_on='Probe Set ID')
        df = df.drop(["features"], axis=1)
    return df


if __name__ == "main":
    # chip
    map_genes = True
    hu6800 = pd.read_csv(os.path.join(DATASET_PATH, "HU6800.chip"), sep='\t')
    X = pd.read_csv(os.path.join(DATASET_PATH, "leukemia_golub_two.csv"), sep=',')
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()

    df_copa = copa_statistic(X=X, y=y, q=0.75, test_class=1)
    df_copa = top_features(X=df_copa, features_name=features_name, map_genes=map_genes)

    df_os = outlier_sum_statistic(X=X, y=y, q=0.75, test_class=1, two_sided_test=False)
    df_os = top_features(X=df_os, features_name=features_name, map_genes=map_genes)

    df_ort = outlier_robust_tstatistic(X=X, y=y, q=0.75, normal_class=0, test_class=1)
    df_ort = top_features(X=df_ort, features_name=features_name, map_genes=map_genes)

    df_heteroiqr = heteroiqr(X=X, y=y)
    df_heteroiqr = top_features(X=df_heteroiqr, features_name=features_name, map_genes=map_genes)
