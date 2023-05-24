import numpy as np
import os
import pandas as pd
import scanpy as sc
import seaborn as sns

from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme(style="white")
METHODS = ["PHet (ΔDispersion)"]
METHODS = ["PHet"]


def train(num_jobs: int = 4):
    # Arguments
    pvalue = 0.01
    sort_by_pvalue = True
    top_k_features = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True
    is_filter = True
    export_spring = True
    num_neighbors = 5
    max_clusters = 10
    feature_metric = "f1"
    cluster_type = "spectral"

    # Models parameters
    bin_pvalues = True
    phet_delta_type = "iqr"
    normalize = "zscore"
    methods_save_name = []
    if phet_delta_type == "hvf":
        temp = "d"
    else:
        temp = "r"
    if bin_pvalues:
        methods_save_name.append("PHet_b" + temp)
    else:
        methods_save_name.append("PHet_nb" + temp)

    # descriptions of the data
    data_name = "plasschaert_mouse"
    suptitle_name = "Basal vs non Basal"
    control_name = "Basal"
    case_name = "non Basal"

    # Exprssion, classes, subtypes, donors, timepoints files
    expression_file_name = data_name + "_matrix.mtx"
    features_file_name = data_name + "_feature_names.csv"
    markers_file = data_name + "_markers.csv"
    classes_file_name = data_name + "_classes.csv"
    subtypes_file = data_name + "_types.csv"
    differential_features_file = data_name + "_limma_features.csv"
    sample_ids_file = data_name + "_library_ids.csv"
    donors_file = data_name + "_donors.csv"
    timepoints_file = data_name + "_timepoints.csv"

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))
    sample_ids = []
    if os.path.exists(os.path.join(DATASET_PATH, sample_ids_file)):
        sample_ids = pd.read_csv(os.path.join(DATASET_PATH, sample_ids_file), sep=',').dropna(axis=1)
        sample_ids = [str(item[0]).lower() for item in sample_ids.values.tolist()]
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

    # Filter data 
    num_examples, num_features = X.shape
    if is_filter:
        example_sums = np.absolute(X).sum(1)
        examples_ids = np.where(example_sums > int(0.01 * num_features))[0]
        X = X[examples_ids]
        y = y[examples_ids]
        subtypes = np.array(subtypes)[examples_ids].tolist()
        if len(sample_ids) != 0:
            sample_ids = np.array(sample_ids)[examples_ids].tolist()
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
        groups.append(["classes"] + [case_name if idx == 1 else control_name for idx in y])
        groups.append(["subtypes"] + subtypes)
        if len(sample_ids) != 0:
            groups.append(["samples"] + sample_ids)
        if len(donors) != 0:
            groups.append(["donors"] + donors)
        if len(timepoints) != 0:
            groups.append(["timepoints"] + timepoints)
        df = pd.DataFrame(groups)
        df.to_csv(os.path.join(RESULT_PATH, data_name + "_groups.csv"), sep=',',
                  index=False, header=False)
        del df

    # Load up/down regulated features
    top_features_true = -1
    if os.path.exists(os.path.join(DATASET_PATH, markers_file)):
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, markers_file)).replace(np.nan, -1)
        top_features_true = list(set([item for item in top_features_true.to_numpy().flatten() if item != -1]))
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
        top_k_features = sum(top_features_true)
    elif os.path.exists(os.path.join(DATASET_PATH, differential_features_file)):
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, differential_features_file), sep=',',
                                        index_col="ID")
        temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
        if top_features_true.shape[1] > 0:
            top_features_true = top_features_true.loc[temp]
            temp = top_features_true[top_features_true["adj.P.Val"] <= pvalue]
            if temp.shape[0] < top_k_features:
                temp = top_features_true[:top_k_features - 1]
                if sort_by_pvalue and temp.shape[0] == 0:
                    plot_topKfeatures = True
            top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:top_k_features]]
        else:
            top_features_true = temp
            top_k_features = len(top_features_true)
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    print("## Perform experimental studies using {0} data...".format(data_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    current_progress = 1
    total_progress = len(METHODS)
    methods_dict = dict()

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[0]))
    estimator = PHeT(normalize=normalize, iqr_range=(25, 75), num_subsamples=1000, delta_type=phet_delta_type,
                     calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                     calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                     weight_range=[0.2, 0.4, 0.8])
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[0]: df})
    current_progress += 1

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
    else:
        print("## Sort features by the score statistic...".format())
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, alpha=pvalue,
                                        X_map=None, map_genes=False, ttest=False)
        else:
            temp = sort_features(X=df, features_name=features_name, X_map=None,
                                 map_genes=False, ttest=False)
        methods_dict[method_name] = temp
    del df

    if top_features_true != -1:
        print("## Scoring results using known regulated features...")
        selected_regulated_features = top_k_features
        temp = np.sum(top_features_true)
        if selected_regulated_features > temp:
            selected_regulated_features = temp
        print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
        list_scores = list()
        for method_idx, item in enumerate(methods_dict.items()):
            if method_idx + 1 == len(METHODS):
                print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / len(METHODS)) * 100,
                                                                        METHODS[method_idx]))
            else:
                print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(METHODS)) * 100,
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
        df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_features_scores.csv"), sep=",")
        print("## Plot barplot using the top {0} features...".format(top_k_features))
        plot_barplot(X=list_scores, methods_name=METHODS, metric="f1", suptitle=suptitle_name,
                     file_name=data_name, save_path=RESULT_PATH)

    temp = np.copy(y)
    temp = temp.astype(str)
    temp[np.where(y == 0)[0]] = control_name
    temp[np.where(y == 1)[0]] = case_name
    y = temp
    list_scores = list()
    score = 0
    print("## Plot UMAP using all features ({0})...".format(num_features))
    score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                      standardize=True, num_neighbors=num_neighbors, min_dist=0, perform_cluster=True,
                      cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                      heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\nAll",
                      file_name=data_name + "_all", save_path=RESULT_PATH)
    list_scores.append(score)
    if top_features_true != -1:
        print("## Plot UMAP using marker features ({0})...".format(sum(top_features_true)))
        temp = np.where(np.array(top_features_true) == 1)[0]
        score = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=features_name, num_features=temp.shape[0],
                          standardize=True, num_neighbors=num_neighbors, min_dist=0, perform_cluster=True,
                          cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                          heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\nMarkers",
                          file_name=data_name + "_markers", save_path=RESULT_PATH)
        list_scores.append(score)

    if plot_topKfeatures:
        print("## Plot UMAP using the top {0} features...".format(top_k_features))
    else:
        print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    method_name), end="\r")
        if plot_topKfeatures:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:top_k_features]]
            temp_feature = [feature for idx, feature in enumerate(features_name) if
                            feature in df['features'].tolist()[:top_k_features]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
            temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
                           standardize=True, num_neighbors=num_neighbors, min_dist=0.0, perform_cluster=True,
                           cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                           heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\n" + method_name,
                           file_name=data_name + "_" + save_name.lower(), save_path=RESULT_PATH)
        df = pd.DataFrame(temp_feature, columns=["features"])
        df.to_csv(os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_features.csv"),
                  sep=',', index=False, header=False)
        if export_spring:
            df = pd.DataFrame(X[:, temp])
            df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_expression.csv"),
                      sep=",", index=False, header=False)
        del df
        list_scores.append(scores)
    index = ["All"]
    if top_features_true != -1:
        index += ["Markers"]
    columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
               "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity",
               "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
    df = pd.DataFrame(list_scores, columns=columns, index=index + METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_cluster_quality.csv"), sep=",")

    print("## Plot bar plot using ARI metric...".format(top_k_features))
    plot_barplot(X=np.array(list_scores)[:, 12], methods_name=index + METHODS, metric="ari",
                 suptitle=suptitle_name, file_name=data_name, save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=8)
