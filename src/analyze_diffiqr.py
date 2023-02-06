import os

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import mmread

from model.deltaiqr import DeltaIQR
from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme(style="white")


def train(num_jobs: int = 4):
    # Arguments
    pvalue = 0.01
    calculate_hstatistic = False
    sort_by_pvalue = True
    topKfeatures = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True
    is_mtx = True

    # 1. Micro-array datasets: allgse412, amlgse2191, bc_ccgse3726, bcca1, bcgse349_350, bladdergse89,
    # braintumor, cmlgse2535, colon, dlbcl, ewsgse967, gastricgse2685, glioblastoma, leukemia_golub,
    # ll_gse1577_2razreda, lung, lunggse1987, meduloblastomigse468, mll, myelodysplastic_mds1,
    # myelodysplastic_mds2, pdac, prostate, prostategse2443, srbct, and tnbc
    # 2. scRNA datasets: camp2, darmanis, lake, yan, camp1, baron, segerstolpe, wang, li, and patel
    # 3. Lung scRNA datasets (mtx): pulseseq, pulseseq_club, pulseseq_club_lineage, pulseseq_goblet, pulseseq_tuft, pulseseq_ionocyte
    # 4. Lung scRNA datasets (csv): plasschaert_human, plasschaert_human_basal_vs_secretory, plasschaert_human_secretory_vs_ciliated, 
    # plasschaert_human_secretory_vs_rare, plasschaert_mouse, plasschaert_mouse_secretory_vs_rare
    file_name = "pulseseq_ionocyte"
    expression_file_name = file_name + "_matrix"
    regulated_features_file = file_name + "_features"
    subtypes_file = file_name + "_types"
    control_name = "Ionocyte"
    case_name = "Tuft"

    # Load expression data
    if not is_mtx:
        X = pd.read_csv(os.path.join(DATASET_PATH, expression_file_name + ".csv"), sep=',').dropna(axis=1)
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
    else:
        features = pd.read_csv(os.path.join(DATASET_PATH, file_name + "_feature_names.csv"), sep=',')
        features_name = features["features"].to_list()
        X = mmread(os.path.join(DATASET_PATH, file_name + "_matrix.mtx"))
        X = X.tocsr().T.toarray()
        y = pd.read_csv(os.path.join(DATASET_PATH, file_name + "_classes.csv"), sep=',')
        y = y["classes"].to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file + ".csv"), sep=',').dropna(axis=1)
    subtypes = [str(idx).lower() for idx in subtypes["subtypes"].to_list()]

    # Filter data based on counts
    num_examples, num_features = X.shape
    example_sums = np.absolute(X).sum(1)
    examples_ids = np.where(example_sums > int(0.01 * num_features))[0]
    X = X[examples_ids]
    y = y[examples_ids]
    num_examples, num_features = X.shape
    del example_sums, examples_ids
    temp = np.absolute(X)
    temp = (temp * 1e6) / temp.sum(axis=1).reshape((num_examples, 1))
    temp[temp > 1] = 1
    temp[temp != 1] = 0
    feature_sums = temp.sum(0)
    del temp
    # if num_examples <= minimum_samples or minimum_samples > num_examples // 2:
    #     minimum_samples = num_examples // 2
    feature_ids = np.where(feature_sums > int(0.01 * num_examples))[0]
    features_name = np.array(features_name)[feature_ids].tolist()
    X = X[:, feature_ids]
    feature_ids = dict([(feature_idx, idx) for idx, feature_idx in enumerate(feature_ids)])
    num_examples, num_features = X.shape
    del feature_sums

    # Load up/down regulated features
    if not is_mtx:
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file + ".csv"), sep=',',
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
    else:
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file + ".csv")).replace(np.nan,
                                                                                                              -1)
        top_features_true = list(set([item for item in top_features_true.to_numpy().flatten() if item != -1]))
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
        topKfeatures = sum(top_features_true)

    print("## Perform experimental studies using {0} data...".format(file_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    current_progress = 1
    total_progress = 2

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "DeltaIQR"), end="\r")
    estimator = DeltaIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_iqr = estimator.fit_predict(X=X, y=y)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "PHet"))
    estimator = PHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=5000, subsampling_size=None,
                     significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                     weight_range=[0.1, 0.3, 0.5], calculate_hstatistic=calculate_hstatistic, num_components=10,
                     num_subclusters=10, binary_clustering=True, calculate_pval=False, num_rounds=50,
                     num_jobs=num_jobs)
    df_phet = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)

    methods_df = dict({"DeltaIQR": df_iqr, "PHet": df_phet})
    methods_name = ["DeltaIQR", "PHet"]

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
    else:
        print("## Sort features by the score statistic...".format())
    for method_idx, item in enumerate(methods_df.items()):
        stat_name, df = item
    method_name = methods_name[method_idx]
    if sort_by_pvalue:
        temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                    X_map=None, map_genes=False, ttest=False)
    else:
        temp = sort_features(X=df, features_name=features_name, X_map=None,
                             map_genes=False, ttest=False)
    df = pd.DataFrame(temp["features"].tolist(), columns=["features"])
    df.to_csv(os.path.join(RESULT_PATH, file_name + "_" + method_name.lower() + "_features.csv"),
              sep=',', index=False)
    methods_df[stat_name] = temp
    del df_iqr, df_phet

    print("## Scoring results using known regulated features...")
    selected_regulated_features = topKfeatures
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for stat_name, df in methods_df.items():
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
    top_features_pred = np.zeros((len(top_features_true)))
    top_features_pred[temp] = 1
    score = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
    list_scores.append(score)
    print("## Plot barplot using the top {0} features...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=list(methods_df.keys()), file_name=file_name,
                 save_path=RESULT_PATH)

    temp = np.copy(y)
    temp = temp.astype(str)
    temp[np.where(y == 0)[0]] = control_name
    temp[np.where(y == 1)[0]] = case_name
    y = temp
    print("## Plot UMAP using all features ({0})...".format(num_features))
    plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features, standardize=True,
              num_neighbors=5, min_dist=0, cluster_type="kmeans", num_clusters=0, max_clusters=10, heatmap_plot=False,
              num_jobs=num_jobs, suptitle=None, file_name=file_name + "_all", save_path=RESULT_PATH)
    print("## Plot UMAP using marker features ({0})...".format(sum(top_features_true)))
    temp = np.where(np.array(top_features_true) == 1)[0]
    plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=features_name, num_features=temp.shape[0],
              standardize=True, num_neighbors=5, min_dist=0, cluster_type="kmeans", num_clusters=0, max_clusters=10,
              heatmap_plot=False, num_jobs=num_jobs, suptitle="UMAP of markers", file_name=file_name + "_markers",
              save_path=RESULT_PATH)
    plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=features_name, num_features=temp.shape[0],
              standardize=True, num_neighbors=5, min_dist=0, perform_cluster=True, cluster_type="kmeans",
              num_clusters=3, max_clusters=10, heatmap_plot=False, num_jobs=num_jobs, suptitle="UMAP of markers",
              file_name=file_name + "_markers", save_path=RESULT_PATH)

    if plot_topKfeatures:
        print("## Plot UMAP using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_df.items()):
        stat_name, df = item
    method_name = methods_name[method_idx]
    if total_progress == method_idx + 1:
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                stat_name))
    else:
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                stat_name), end="\r")
    if plot_topKfeatures:
        temp = [idx for idx, feature in enumerate(features_name) if
                feature in df['features'].tolist()[:topKfeatures]]
        temp_feature = [feature for idx, feature in enumerate(features_name) if
                        feature in df['features'].tolist()[:topKfeatures]]
    else:
        temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
    num_features = len(temp)
    plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
              standardize=True, num_neighbors=5, min_dist=0.0, perform_cluster=True, cluster_type="kmeans",
              num_clusters=3, max_clusters=10, heatmap_plot=False, num_jobs=num_jobs, suptitle=stat_name.upper(),
              file_name=file_name + "_" + method_name.lower(), save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
