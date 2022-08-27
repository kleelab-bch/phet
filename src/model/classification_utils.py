import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC

EPSILON = np.finfo(np.float).eps

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
sns.set_theme()

plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=16)


def create_directory(name_folder: str = "test", save_path: str = "."):
    file_path = os.path.join(save_path, name_folder)
    if os.path.exists(save_path):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        os.makedirs(file_path)
        return file_path
    else:
        raise Exception("Invalid file path: {0}".format(save_path))


def features_selection(X, attribute_columns, variance_threshold, verbose: bool = False):
    if verbose:
        print("\t    >> Selecting features...")
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.feature_names_in_ = attribute_columns
    X = selector.fit_transform(X)
    attribute_columns = [attribute_columns[idx]
                         for idx, feat in enumerate(selector.get_support()) if feat]
    return X, attribute_columns


def SVM_RFE(X, y, num_features: int, standardize: bool = True):
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

    Returns
    -------
    SVM-RFE results for comparison to Hetero-Net
    """

    # get feature names
    feature_names = X.columns
    # standardize if needed
    if standardize:
        X = zscore(X)

    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=num_features, step=1)
    selector = selector.fit(X, y)

    # reindex based on ranking
    feature_names = pd.Series(feature_names)
    feature_names.index = selector.ranking_

    rfe_results = pd.DataFrame(list(feature_names.sort_index().reset_index(drop=True)))
    rfe_results.columns = ['features']

    return rfe_results
