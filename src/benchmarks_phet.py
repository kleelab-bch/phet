import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme(style="white")


def train(num_jobs: int = 4):
    # Arguments
    feature_weight = [0.4, 0.3, 0.2, 0.1]
    weight_range = [0.1, 0.4, 0.8]
    pvalue = 0.01
    sort_by_pvalue = True
    topKfeatures = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True
    is_filter = True
    max_clusters = 10
    cluster_type = "kmeans"
    export_spring = False
    methods = ["PHet (no Binning)", "PHet"]
    methods_save_name = ["PHet_nb", "PHet_b"]

    # descriptions of the data
    file_name = "her2_positive"
    suptitle_name = "Tuft vs Ionocyte"
    control_name = "Tuft"
    case_name = "Ionocyte"

    # Exprssion, classes, subtypes, donors, timepoints Files
    expression_file_name = file_name + "_matrix.mtx"
    features_file_name = file_name + "_feature_names.csv"
    markers_file = file_name + "_markers.csv"
    classes_file_name = file_name + "_classes.csv"
    subtypes_file = file_name + "_types.csv"
    differential_features_file = file_name + "_diff_features.csv"
    donors_file = file_name + "_donors.csv"
    timepoints_file = file_name + "_timepoints.csv"

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))
    donors = []
    if os.path.exists(os.path.join(DATASET_PATH, donors_file)):
        donors = pd.read_csv(os.path.join(DATASET_PATH, donors_file), sep=',').dropna(axis=1)
        donors = [str(item[0]).lower() for item in donors.values.tolist()]
    timepoints = []
    if os.path.exists(os.path.join(DATASET_PATH, timepoints_file)):
        timepoints = pd.read_csv(os.path.join(DATASET_PATH, timepoints_file), sep=',').dropna(axis=1)
        timepoints = [str(item[0]).lower() for item in timepoints.values.tolist()]

    # Load features, expression, and class data
    features_name = pd.read_csv(os.path.join(DATASET_PATH, features_file_name), sep=',')
    features_name = features_name["features"].to_list()
    y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter data based on counts
    num_examples, num_features = X.shape
    if is_filter:
        example_sums = np.absolute(X).sum(1)
        examples_ids = np.where(example_sums > int(0.01 * num_features))[0]
        X = X[examples_ids]
        y = y[examples_ids]
        subtypes = np.array(subtypes)[examples_ids].tolist()
        if len(donors) != 0:
            donors = np.array(donors)[examples_ids].tolist()
        if len(timepoints) != 0:
            timepoints = np.array(timepoints)[examples_ids].tolist()
        num_examples, num_features = X.shape
        del example_sums, examples_ids
        temp = np.absolute(X)
        temp = (temp * 1e6) / temp.sum(axis=1).reshape((num_examples, 1))
        temp[temp > 1] = 1
        temp[temp != 1] = 0
        feature_sums = temp.sum(0)
        del temp
        feature_ids = np.where(feature_sums > int(0.01 * num_examples))[0]
        features_name = np.array(features_name)[feature_ids].tolist()
        X = X[:, feature_ids]
        num_examples, num_features = X.shape
        del feature_sums

    # Save subtypes for SPRING
    if export_spring:
        groups = []
        groups.append(["subtypes"] + subtypes)
        if len(donors) != 0:
            groups.append(["donors"] + donors)
        if len(timepoints) != 0:
            groups.append(["timepoints"] + timepoints)
        df = pd.DataFrame(groups)
        df.to_csv(os.path.join(RESULT_PATH, file_name + "_groups.csv"), sep=',',
                  index=False, header=False)
        del df

    # Load up/down regulated features
    top_features_true = -1
    if os.path.exists(os.path.join(DATASET_PATH, markers_file)):
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, markers_file)).replace(np.nan, -1)
        top_features_true = list(set([item for item in top_features_true.to_numpy().flatten() if item != -1]))
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
        topKfeatures = sum(top_features_true)
    elif os.path.exists(os.path.join(DATASET_PATH, differential_features_file)):
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

    print("## Perform experimental studies using {0} data...".format(file_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    current_progress = 1
    total_progress = len(methods)
    methods_dict = dict()

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            methods[0]), end="\r")
    estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                     calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=True,
                     bin_KS_pvalues=False, feature_weight=feature_weight, weight_range=weight_range)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({methods[0]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, methods[1]))
    estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                     calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=True,
                     bin_KS_pvalues=True, feature_weight=feature_weight, weight_range=weight_range)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({methods[1]: df})

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
    else:
        print("## Sort features by the score statistic...".format())
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = methods[method_idx]
        save_name = methods_save_name[method_idx]
        if sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                        X_map=None, map_genes=False, ttest=False)
        else:
            temp = sort_features(X=df, features_name=features_name, X_map=None,
                                 map_genes=False, ttest=False)
        methods_dict[method_name] = temp
    del df

    if top_features_true != -1:
        print("## Scoring results using known regulated features...")
        selected_regulated_features = topKfeatures
        temp = np.sum(top_features_true)
        if selected_regulated_features > temp:
            selected_regulated_features = temp
        print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
        list_scores = list()
        for method_idx, item in enumerate(methods_dict.items()):
            if method_idx + 1 == len(methods):
                print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / len(methods)) * 100,
                                                                          methods[method_idx]))
            else:
                print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(methods)) * 100,
                                                                          methods[method_idx]), end="\r")
            method_name, df = item
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:selected_regulated_features].tolist()]
            top_features_pred = np.zeros((len(top_features_true)))
            top_features_pred[temp] = 1
            score = comparative_score(pred_features=top_features_pred, true_features=top_features_true, metric="f1")
            list_scores.append(score)

        df = pd.DataFrame(list_scores, columns=["Scores"], index=methods)
        df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_features_scores.csv"), sep=",")
        print("## Plot barplot using the top {0} features...".format(topKfeatures))
        plot_barplot(X=list_scores, methods_name=methods, metric="f1", suptitle=suptitle_name,
                     file_name=file_name, save_path=RESULT_PATH)

    temp = np.copy(y)
    temp = temp.astype(str)
    temp[np.where(y == 0)[0]] = control_name
    temp[np.where(y == 1)[0]] = case_name
    y = temp
    list_scores = list()
    score = 0
    print("## Plot UMAP using all features ({0})...".format(num_features))
    score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                      standardize=True, num_neighbors=5, min_dist=0, perform_cluster=True, cluster_type=cluster_type,
                      num_clusters=num_clusters, max_clusters=max_clusters, apply_hungarian=False, heatmap_plot=False,
                      num_jobs=num_jobs, suptitle=suptitle_name + "\nAll", file_name=file_name + "_all",
                      save_path=RESULT_PATH)
    list_scores.append(score)
    if top_features_true != -1:
        print("## Plot UMAP using marker features ({0})...".format(sum(top_features_true)))
        temp = np.where(np.array(top_features_true) == 1)[0]
        score = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=features_name, num_features=temp.shape[0],
                          standardize=True, num_neighbors=5, min_dist=0, perform_cluster=True,
                          cluster_type=cluster_type,
                          num_clusters=num_clusters, max_clusters=max_clusters, apply_hungarian=False,
                          heatmap_plot=False,
                          num_jobs=num_jobs, suptitle=suptitle_name + "\nMarkers", file_name=file_name + "_markers",
                          save_path=RESULT_PATH)
        list_scores.append(score)

    if plot_topKfeatures:
        print("## Plot UMAP using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = methods[method_idx]
        save_name = methods_save_name[method_idx]
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
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
                          standardize=True, num_neighbors=5, min_dist=0.0, perform_cluster=True,
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
    index = ["All"]
    if top_features_true != -1:
        index += ["Markers"]
    df = pd.DataFrame(list_scores, columns=["Scores"], index=index + methods)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_cluster_quality.csv"), sep=",")

    print("## Plot barplot using to demonstrate clustering accuracy...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=index + methods, metric="ari",
                 suptitle=suptitle_name, file_name=file_name, save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=8)
