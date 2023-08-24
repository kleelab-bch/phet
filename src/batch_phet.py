import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap
from utility.utils import clustering
from utility.utils import comparative_score
from utility.utils import significant_features

sns.set_theme(style="white")


def train(num_jobs: int = 4):
    # Arguments
    alpha = 0.01
    top_k_features = 100
    is_filter = True
    method_name = "PHet"
    save_name = "PHet_br"

    # Descriptions of the data
    data_name = "baron1"
    suptitle_name = "Baron"
    control_name = "0"
    case_name = "1"

    # Models parameters
    ari_threshold = 0.7
    num_epochs = 1
    search_optimum_f1 = False
    cluster_type = "spectral"
    standardize = True

    # Exprssion, classes, subtypes, donors, timepoints files
    expression_file_name = data_name + "_matrix.mtx"
    features_file_name = data_name + "_feature_names.csv"
    markers_file = data_name + "_markers.csv"
    classes_file_name = data_name + "_classes.csv"
    subtypes_file = data_name + "_types.csv"
    differential_features_file = data_name + "_limma_features.csv"

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
    if is_filter:
        example_sums = np.absolute(X).sum(1)
        examples_ids = np.where(example_sums > int(0.01 * num_features))[0]
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
        del temp
        feature_ids = np.where(feature_sums > int(0.01 * num_examples))[0]
        features_name = np.array(features_name)[feature_ids].tolist()
        X = X[:, feature_ids]
        num_examples, num_features = X.shape
        del feature_sums

    # True cluster labels
    labels_true = np.unique(subtypes)
    labels_true = dict([(item, idx) for idx, item in enumerate(labels_true)])
    labels_true = [labels_true[item] for item in subtypes]

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
            temp = top_features_true[top_features_true["adj.P.Val"] < alpha]
            if temp.shape[0] < top_k_features:
                temp = top_features_true[:top_k_features - 1]
            top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:top_k_features]]
        else:
            top_features_true = temp
            top_k_features = len(top_features_true)
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
    selected_regulated_features = top_k_features
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp

    print("## Perform experimental studies using {0} data...".format(suptitle_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    epoch = 0
    curr_ari = 0.0
    opt_ari = 0.0
    curr_f1 = 0.0
    opt_f1 = 0.0
    optimum_feature = []
    optimum_labels_pred = []
    while epoch < num_epochs and curr_ari < ari_threshold:
        epoch += 1
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = significant_features(X=df, features_name=features_name, alpha=alpha,
                                  X_map=None, map_genes=False, ttest=False)
        temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        labels_pred = clustering(X=X[:, temp], cluster_type=cluster_type, num_clusters=num_clusters,
                                 num_jobs=num_jobs, predict=True)
        curr_ari = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)

        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'].tolist()[:selected_regulated_features]]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        curr_f1 = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                    metric="f1")
        if opt_ari <= curr_ari:
            if search_optimum_f1:
                if epoch == 1 or opt_f1 > curr_f1:
                    continue
            if opt_ari == curr_ari:
                if opt_f1 > curr_f1:
                    continue
            opt_ari = curr_ari
            optimum_feature = df['features'].tolist()
            optimum_labels_pred = labels_pred
            opt_f1 = curr_f1

        if epoch == num_epochs:
            print("\t >> Epoch: {0} (of {1}); ARI: {2:.4f} ({3:.4f}); F1: {4:.4f} ({5:.4f})".format(epoch, num_epochs,
                                                                                                    curr_ari, opt_ari,
                                                                                                    curr_f1, opt_f1))
        else:
            print("\t >> Epoch: {0} (of {1}); ARI: {2:.4f} ({3:.4f}); F1: {4:.4f} ({5:.4f})".format(epoch, num_epochs,
                                                                                                    curr_ari, opt_ari,
                                                                                                    curr_f1, opt_f1),
                  end="\r")

    print("\n## Optimum ARI: {0:.4f}; Optimum F1: {1:.4f}".format(opt_ari, opt_f1))

    # Store the optimum feature F1 score
    temp = [idx for idx, feature in enumerate(features_name) if
            feature in optimum_feature[:selected_regulated_features]]
    top_features_pred = np.zeros((len(top_features_true)))
    top_features_pred[temp] = 1
    f1_score = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
    df = pd.DataFrame([f1_score], columns=["Scores"], index=[method_name])
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_features_scores_phet.csv"), sep=",")

    # Store the optimum feature set clustering quality and plots
    temp = [idx for idx, feature in enumerate(features_name) if feature in optimum_feature]
    num_features = len(temp)
    y = y.astype(str)
    y[np.where(y == 0)[0]] = control_name
    y[np.where(y == 1)[0]] = case_name
    list_scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=[], num_features=num_features,
                            labels_pred=optimum_labels_pred, standardize=standardize, num_neighbors=5,
                            min_dist=0.0, perform_cluster=True, cluster_type=cluster_type, num_clusters=num_clusters,
                            max_clusters=10, heatmap_plot=False, num_jobs=num_jobs,
                            suptitle=suptitle_name + "\n" + method_name,
                            file_name=data_name + "_" + save_name.lower(), save_path=RESULT_PATH)
    columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
               "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity",
               "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
    df = pd.DataFrame(np.array(list_scores)[None, :], columns=columns, index=[method_name])
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_cluster_quality_phet.csv"), sep=",")

    # Store the optimum feature set
    df = pd.DataFrame(optimum_feature, columns=["features"])
    df.to_csv(os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_features.csv"),
              sep=',', index=False, header=False)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=8)
