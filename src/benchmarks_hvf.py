import os
from copy import deepcopy

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from model.deltahvfmean import DeltaHVFMean
from model.deltaiqrmean import DeltaIQRMean
from model.hvf import SeuratHVF, HIQR
from model.nonparametric_test import StudentTTest, WilcoxonRankSumTest
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["HVF (composite)", "HVF (by condition)", "ΔHVF", "ΔHVF+ΔMean"]


def train(num_jobs: int = 4):
    # Arguments
    minimum_samples = 5
    pvalue = 0.01
    sort_by_pvalue = True
    export_spring = False
    topKfeatures = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True
    num_neighbors = 5
    max_clusters = 10
    feature_metric = "f1"
    log_transform = False
    cluster_type = "spectral"
    methods_save_name = ["hvf_a", "hvf_c", "deltahvf", "deltahvfmean"]
    # Descriptions of the data
    file_name = "patel"
    suptitle_name = "Patel"

    # Exprssion, classes, subtypes, donors, timepoints Files
    expression_file_name = file_name + "_matrix.mtx"
    features_file_name = file_name + "_feature_names.csv"
    classes_file_name = file_name + "_classes.csv"
    subtypes_file = file_name + "_types.csv"
    differential_features_file = file_name + "_limma_features.csv"

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
    num_examples, num_features = X.shape
    del temp, feature_sums

    # Save subtypes for SPRING
    if export_spring:
        df = pd.DataFrame(subtypes, columns=["subtypes"]).T
        df.to_csv(os.path.join(RESULT_PATH, file_name + "_subtypes.csv"), sep=',', header=False)
        del df

    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, differential_features_file), sep=',',
                                    index_col="ID")
    temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
    if top_features_true.shape[1] > 0:
        top_features_true = top_features_true.loc[temp]
        temp = top_features_true[top_features_true["adj.P.Val"] <= pvalue]
        if temp.shape[0] < topKfeatures:
            temp = top_features_true[:topKfeatures - 1]
            if sort_by_pvalue and temp.shape[0] == 0:
                plot_topKfeatures = True
        top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:topKfeatures]]
    else:
        top_features_true = temp
        topKfeatures = len(top_features_true)
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    print("## Perform experimental studies using {0} data...".format(suptitle_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    current_progress = 1
    total_progress = len(METHODS)
    methods_dict = dict()

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[0]), end="\r")
    estimator = SeuratHVF(per_condition=False, log_transform=log_transform, 
                          num_top_features=num_features, min_disp=0.5, 
                          min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    df = estimator.fit_predict(X=temp_X, y=y)
    del temp_X
    methods_dict.update({METHODS[0]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[1]), end="\r")
    estimator = SeuratHVF(per_condition=True, log_transform=log_transform, 
                          num_top_features=num_features, min_disp=0.5, 
                          min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    df = estimator.fit_predict(X=temp_X, y=y)
    del temp_X
    methods_dict.update({METHODS[1]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[2]), end="\r")
    estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=log_transform, 
                             num_top_features=num_features, min_disp=0.5,
                             min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    df = estimator.fit_predict(X=temp_X, y=y)
    del temp_X
    methods_dict.update({METHODS[2]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                            METHODS[3]))
    estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=log_transform, 
                             num_top_features=num_features, min_disp=0.5,
                             min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    df = estimator.fit_predict(X=temp_X, y=y)
    del temp_X
    methods_dict.update({METHODS[3]: df})

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
    else:
        print("## Sort features by the score statistic...".format())
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if method_name in ['DECO', 't-statistic', 'Wilcoxon', 'LIMMA']:
            continue
        if sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                        X_map=None, map_genes=False, ttest=False)
        else:
            temp = sort_features(X=df, features_name=features_name, X_map=None,
                                 map_genes=False, ttest=False)
        methods_dict[method_name] = temp

    print("## Scoring results using up/down regulated features...")
    selected_regulated_features = topKfeatures
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for method_idx, item in enumerate(methods_dict.items()):
        if method_idx + 1 == len(METHODS):
            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / len(METHODS)) * 100,
                                                                      METHODS[method_idx]))
        else:
            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((method_idx / len(METHODS)) * 100,
                                                                      METHODS[method_idx]), end="\r")
        method_name, df = item
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        score = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                  metric=feature_metric)
        list_scores.append(score)

    df = pd.DataFrame(list_scores, columns=["Scores"], index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_features_scores.csv"), sep=",")
    print("## Plot barplot using the top {0} features...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=METHODS, metric=feature_metric, suptitle=suptitle_name,
                 file_name=file_name, save_path=RESULT_PATH)

    list_scores = [0]
    if plot_topKfeatures:
        print("## Plot UMAP using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name), end="\r")
        if plot_topKfeatures:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:topKfeatures]]
            temp_feature = [feature for idx, feature in enumerate(features_name) if
                            feature in df['features'].tolist()[:topKfeatures]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
            temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        score = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
                          standardize=True, num_neighbors=num_neighbors, min_dist=0.0, perform_cluster=True,
                          cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                          apply_hungarian=False, heatmap_plot=False, num_jobs=num_jobs,
                          suptitle=suptitle_name + "\n" + method_name, file_name=file_name + "_" + save_name.lower(),
                          save_path=RESULT_PATH)
        df = pd.DataFrame(temp_feature, columns=["features"])
        df.to_csv(os.path.join(RESULT_PATH, file_name + "_" + save_name.lower() + "_features.csv"),
                  sep=',', index=False, header=False)
        if export_spring:
            df = pd.DataFrame(X[:, temp])
            df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_" + save_name.lower() + "_expression.csv"),
                      sep=",", index=False, header=False)
        del df
        list_scores.append(score)

    df = pd.DataFrame(list_scores, columns=["Scores"], index=["All"] + METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_cluster_quality.csv"), sep=",")

    print("## Plot bar plot using to demonstrate clustering accuracy...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=["All"] + METHODS, metric="ari",
                 suptitle=suptitle_name, file_name=file_name, save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)
