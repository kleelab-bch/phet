import os
from copy import deepcopy

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import zscore

from model.deltaiqrmean import DeltaIQRMean
from model.hvf import SeuratHVF
from model.nonparametric_test import ZTest
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap
from utility.utils import sort_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["DE", "IQR", "HD", "HF"]


def train(num_jobs: int = 4):
    # Filtering arguments
    minimum_samples = 5

    # Models parameters
    direction = "both"
    methods_save_name = ["de", "iqr", "hd", "hvf"]
    adjusted_alpha = 0.01
    de_alpha = 0.01
    iqr_distance = 1
    min_disp = 0.5

    # Clustering and UMAP parameters
    num_neighbors = 5
    max_clusters = 10
    cluster_type = "spectral"

    # Descriptions of the data
    data_name = "baron1"
    suptitle_name = "Baron"
    log_transform = False
    exponentiate = False
    standardize = False
    perform_undersampling = True
    palette = None
    if data_name == "baron1":
        palette = {'beta': '#1f77b4', 'delta': '#ff7f0e', 'ductal': '#2ca02c',
                   'alpha': '#d62728', 'gamma': '#9467bd', 'endothelial': '#8c564b',
                   'macrophage': '#e377c2', 'schwann': '#7f7f7f', 't_cell': '#bcbd22'}

    # Expression, classes, subtypes, donors, timepoints files
    expression_file_name = data_name + "_matrix.mtx"
    features_file_name = data_name + "_feature_names.csv"
    classes_file_name = data_name + "_classes.csv"
    subtypes_file = data_name + "_types.csv"

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))

    # Load features, expression, and class data
    features_name = pd.read_csv(os.path.join(DATASET_PATH, features_file_name), sep=',')
    features_name = features_name["features"].to_list()
    y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter data
    num_examples, num_features = X.shape
    example_sums = np.absolute(X).sum(1)
    examples_ids = np.where(example_sums >= 5)[0]  # filter out cells below 5
    X = X[examples_ids]
    y = y[examples_ids]
    subtypes = np.array(subtypes)[examples_ids].tolist()
    num_examples, num_features = X.shape
    del example_sums, examples_ids
    temp = np.absolute(X)
    temp = (temp * 1e6) / temp.sum(axis=1).reshape((num_examples, 1))
    temp[temp > 1] = 1
    temp[temp != 1] = 0
    feature_sums = temp.sum(0)
    if num_examples <= minimum_samples or minimum_samples > num_examples // 2:
        minimum_samples = num_examples // 2
    feature_ids = np.where(feature_sums >= minimum_samples)[0]
    features_name = np.array(features_name)[feature_ids].tolist()
    X = X[:, feature_ids]
    feature_ids = dict([(feature_idx, idx) for idx, feature_idx in enumerate(feature_ids)])
    num_examples, num_features = X.shape
    del temp, feature_sums

    print("## Perform experimental studies using {0} data...".format(suptitle_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    current_progress = 1
    total_progress = len(METHODS)
    methods_dict = dict()

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[0]), end="\r")
    estimator = ZTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                      adjusted_alpha=adjusted_alpha)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                       ttest=False, ascending=True)
    df = df[df["score"] < de_alpha]
    methods_dict.update({METHODS[0]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[1]), end="\r")
    estimator = DeltaIQRMean(calculate_deltamean=False, normalize="zscore", iqr_range=(25, 75))
    df = estimator.fit_predict(X=X, y=y)
    df = df[:, 3]
    df = np.absolute(zscore(df))
    df = df[:, None]
    df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                       ttest=False, ascending=False)
    df = df[df["score"] > iqr_distance]
    methods_dict.update({METHODS[1]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[2]), end="\r")
    temp = list(set(methods_dict[METHODS[0]]["features"].to_list()).intersection(
        methods_dict[METHODS[1]]["features"].to_list()))
    df = pd.DataFrame([temp, range(len(temp))]).T
    df.columns = ["features", "score"]
    methods_dict.update({METHODS[2]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[3]))
    estimator = SeuratHVF(per_condition=False, log_transform=log_transform,
                          num_top_features=None, min_disp=min_disp,
                          min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    if exponentiate:
        temp_X = np.exp(temp_X)
    df = estimator.fit_predict(X=temp_X, y=y)
    del temp_X
    df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                       ttest=False, ascending=False)
    df = df[df["score"] > 0.0]
    methods_dict.update({METHODS[3]: df})

    list_scores = list()
    print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        # if method_name == "DE":
        #     continue
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name), end="\r")
        temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        if num_features == 0:
            temp = [idx for idx, feature in enumerate(features_name)]
            temp_feature = [feature for idx, feature in enumerate(features_name)]
        scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
                           perform_undersampling=perform_undersampling, standardize=standardize,
                           num_neighbors=num_neighbors,
                           min_dist=0.0, perform_cluster=True, cluster_type=cluster_type, num_clusters=num_clusters,
                           max_clusters=max_clusters, heatmap_plot=False, palette=palette, num_jobs=num_jobs,
                           suptitle=suptitle_name + "\n" + method_name, file_name=data_name + "_" + save_name.lower(),
                           save_path=RESULT_PATH)
        df = pd.DataFrame(temp_feature, columns=["features"])
        df.to_csv(os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_features.csv"),
                  sep=',', index=False, header=False)
        del df
        list_scores.append(scores)

    columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
               "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity",
               "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
    df = pd.DataFrame(list_scores, columns=columns, index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_cluster_quality.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)
