import os

import numpy as np
import pandas as pd
import seaborn as sns

from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)


def train(num_jobs: int = 4):
    # Arguments
    minimum_samples = 5
    pvalue = 0.01
    sort_by_pvalue = True
    topKfeatures = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True
    max_clusters = 10
    cluster_type = "kmeans"
    # cluster_type = "spectral"
    methods_name = ["t-statistic", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "ΔIQR", "PHet"]

    # 1. Microarray datasets: allgse412, amlgse2191, bc_ccgse3726, bcca1, bcgse349_350, bladdergse89,
    # braintumor, cmlgse2535, colon, dlbcl, ewsgse967, gastricgse2685, glioblastoma, leukemia_golub, 
    # ll_gse1577_2razreda, lung, lunggse1987, meduloblastomigse468, mll, myelodysplastic_mds1, 
    # myelodysplastic_mds2, pdac, prostate, prostategse2443, srbct, and tnbc
    # 2. scRNA datasets: camp2, darmanis, lake, yan, camp1, baron, segerstolpe, wang, li, and patel
    ### For the paper: 
    # 1. Microarray datasets: allgse412, bc_ccgse3726, bladdergse89, braintumor, gastricgse2685, glioblastoma, 
    # leukemia_golub, lung, lunggse1987, mll, srbct
    # 2. scRNA datasets: darmanis, yan, camp1, baron, li, and patel
    file_name = "srbct"
    data_name = "SRBCT"
    expression_file_name = file_name + "_matrix"
    regulated_features_file = file_name + "_features"
    subtypes_file = file_name + "_types"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, expression_file_name + ".csv"), sep=',').dropna(axis=1)
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file + ".csv"), sep=',').dropna(axis=1)
    subtypes = subtypes["subtypes"].to_list()
    num_clusters = len(np.unique(subtypes))

    # Filter data based on counts (CPM)
    example_sums = np.absolute(X).sum(1)
    examples_ids = np.where(example_sums >= 5)[0]
    X = X[examples_ids]
    y = y[examples_ids]
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

    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file + ".csv"), sep=',',
                                    index_col="ID")
    temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
    top_features_true = top_features_true.loc[temp]
    temp = top_features_true[top_features_true["adj.P.Val"] <= pvalue]
    if temp.shape[0] < topKfeatures:
        temp = top_features_true[:topKfeatures - 1]
        if sort_by_pvalue and temp.shape[0] == 0:
            plot_topKfeatures = True
    top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:topKfeatures]]
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    print("## Perform experimental studies using {0} data...".format(data_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    total_progress = len(methods_name)

    result_path = os.path.join(RESULT_PATH, "microarray")
    feature_files = [file_name + "_" + method.lower() + "_features.csv" for method in methods_name]
    feature_files[0] = file_name + "_ttest_features.csv"
    feature_files[7] = file_name + "_deco.csv"
    feature_files[8] = file_name + "_deltaiqr_features.csv"
    methods_dict = dict()
    for idx, f in enumerate(feature_files):
        df = pd.read_csv(os.path.join(result_path, f), sep=',')
        if methods_name[idx] == "DECO":
            df = [(features_name[feature_ids[int(df[1][0])]], df[1][1]) for df in df.iterrows()]
            df = pd.DataFrame(df, columns=["features", "score"])
        methods_dict.update({methods_name[idx]: df})

    print("## Scoring results using known regulated features...")
    selected_regulated_features = topKfeatures
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for method_idx, item in enumerate(methods_dict.items()):
        if method_idx + 1 == len(methods_name):
            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / len(methods_name)) * 100,
                                                                      methods_name[method_idx]))
        else:
            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(methods_name)) * 100,
                                                                      methods_name[method_idx]), end="\r")
        method_name, df = item
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        score = comparative_score(pred_features=top_features_pred, true_features=top_features_true, metric="f1")
        list_scores.append(score)

    df = pd.DataFrame(list_scores, columns=["Scores"], index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_features_scores.csv"), sep=",")
    print("## Plot barplot using the top {0} features...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=methods_name, metric="f1", suptitle=data_name,
                 file_name=file_name, save_path=RESULT_PATH)

    list_scores = list()
    print("## Plot UMAP using all features ({0})...".format(num_features))
    score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                      standardize=True, num_neighbors=5, min_dist=0, perform_cluster=True, cluster_type=cluster_type,
                      num_clusters=num_clusters, max_clusters=max_clusters, apply_hungarian=False, heatmap_plot=False,
                      num_jobs=num_jobs, suptitle=data_name + ": ", file_name=file_name + "_all", save_path=RESULT_PATH)
    list_scores.append(score)

    if plot_topKfeatures:
        print("## Plot UMAP using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = methods_name[method_idx]
        save_name = method_name
        if method_name == "t-statistic":
            save_name = "ttest"
        elif method_name == "ΔIQR":
            save_name = "DeltaIQR"
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
                          suptitle=data_name + ": " + method_name, file_name=file_name + "_" + save_name.lower(),
                          save_path=RESULT_PATH)
        list_scores.append(score)

    df = pd.DataFrame(list_scores, columns=["Scores"], index=["All"] + methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_cluster_quality.csv"), sep=",")

    print("## Plot barplot using to demonstrate clustering accuracy...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=["All"] + methods_name, metric="ari",
                 suptitle=data_name, file_name=file_name, save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)
