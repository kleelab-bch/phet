import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from model.phet import PHeT
from sklearn.metrics import adjusted_rand_score
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap
from utility.utils import comparative_score
from utility.utils import significant_features
from utility.utils import clustering

sns.set_theme(style="white")
METHODS = ["PHet"]


def train(num_jobs: int = 4):
    # Arguments
    pvalue = 0.01
    topKfeatures = 100
    is_filter = True
    num_neighbors = 5
    max_clusters = 10
    feature_metric = "f1"

    # Models parameters
    methods_save_name = ["PHet_br"]
    cluster_type = "kmeans"
    ari_threshold = 0.9
    num_epochs = 50

    # descriptions of the data
    data_name = "srbct"
    suptitle_name = "SRBCT"
    control_name = "0"
    case_name = "1"

    # Exprssion, classes, subtypes, donors, timepoints files
    expression_file_name = data_name + "_matrix.mtx"
    features_file_name = data_name + "_feature_names.csv"
    markers_file = data_name + "_markers.csv"
    classes_file_name = data_name + "_classes.csv"
    subtypes_file = data_name + "_types.csv"
    differential_features_file = data_name + "_limma_features.csv"
    donors_file = data_name + "_donors.csv"
    timepoints_file = data_name + "_timepoints.csv"

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

    # Filter data 
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
            top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:topKfeatures]]
        else:
            top_features_true = temp
            topKfeatures = len(top_features_true)
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    print("## Perform experimental studies using {0} data...".format(data_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    selected_regulated_features = topKfeatures
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp

    epoch = 0
    ari_score = 0.0
    optimum_ari = 0.0
    optimum_feature = []
    while epoch < num_epochs and ari_score < ari_threshold:
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        epoch += 1
        method_name = METHODS[0]
        save_name = methods_save_name[0]
        temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                    X_map=None, map_genes=False, ttest=False)
        df = temp
        temp = np.copy(y)
        temp = temp.astype(str)
        temp[np.where(y == 0)[0]] = control_name
        temp[np.where(y == 1)[0]] = case_name
        temp_y = temp
        temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        labels_pred = clustering(X=X[:, temp], cluster_type=cluster_type, num_clusters=num_clusters,
                                 num_jobs=num_jobs, predict=True)
        labels_true = np.array(labels_true)
        labels_pred = np.array(labels_pred)
        ari_score = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)
        if ari_score > optimum_ari:
            optimum_ari = ari_score
            optimum_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        print("\t >> Epoch: {0}; ARI: {1:.4f} ({2:.4f})".format(epoch, ari_score, optimum_ari), end="\r")

    # Store the optimum feature set
    df = pd.DataFrame(optimum_feature, columns=["features"])
    df.to_csv(os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_features.csv"),
                  sep=',', index=False, header=False)
    # Store the optimum feature F1 score
    temp = [idx for idx, feature in enumerate(features_name) if feature in optimum_feature[:selected_regulated_features].tolist()]
    top_features_pred = np.zeros((len(top_features_true)))
    top_features_pred[temp] = 1
    f1_score = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric=feature_metric)
    df = pd.DataFrame([f1_score], columns=["Scores"], index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_features_scores_phet.csv"), sep=",")

    # Store the optimum feature set clustering quality and plots
    list_scores = plot_umap(X=X[:, temp], y=temp_y, subtypes=subtypes, features_name=optimum_feature,
                                num_features=num_features, standardize=True, num_neighbors=num_neighbors, min_dist=0.0,
                                perform_cluster=True, cluster_type=cluster_type, num_clusters=num_clusters,
                                max_clusters=max_clusters, heatmap_plot=False, num_jobs=num_jobs,
                                suptitle=suptitle_name + "\n" + method_name,
                                file_name=data_name + "_" + save_name.lower(), save_path=RESULT_PATH)
    columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
               "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity", 
               "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
    df = pd.DataFrame(np.array(list_scores)[None, :], columns=columns, index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_cluster_quality_phet.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=8)
