import os

import numpy as np
import pandas as pd
import seaborn as sns

from model.copa import COPA
from model.dids import DIDS
from model.globulin import GLOBULIN
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import sort_features, comparative_score

sns.set_theme()


def train(num_jobs: int = 4):
    # Arguments
    top_k_features = 100
    direction = "both"

    # Datasets:
    # 1. myelodysplastic_mds1_matrix and myelodysplastic_mds1_features
    # 2. myelodysplastic_mds2_matrix and myelodysplastic_mds2_features
    # 3. bcca1_matrix and bcca1_features
    # 4. leukemia_golub_matrix and leukemia_golub_features
    # 5. colon_matrix and colon_features
    file_name = "bcca1_matrix"
    regulated_features_file = "bcca1_features"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), sep=',').dropna(axis=1)
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file + ".csv"), sep=',')
    top_features_true = [str(feature_idx) for feature_idx in top_features_true["ID"].to_list()[:top_k_features]]
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    print("## Perform experimental studies using {0} data...".format(file_name))
    methods_name = ["COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "UHet_zscore", "UHet_robust", "GLOBULIN"]
    current_progress = 1
    total_progress = len(methods_name)


    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (zscore)"), end="\r")
    estimator = UHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_z = estimator.fit_predict(X=X, y=y)
    df_uhet_z = sort_features(X=df_uhet_z, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "COPA"), end="\r")
    estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
    df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_copa = sort_features(X=df_copa, features_name=features_name, X_map=None, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "OS"), end="\r")
    estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                    calculate_pval=False)
    df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_os = sort_features(X=df_os, features_name=features_name, X_map=None, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "ORT"), end="\r")
    estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                       calculate_pval=False)
    df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_ort = sort_features(X=df_ort, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "MOST"), end="\r")
    estimator = MOST(direction=direction, calculate_pval=False)
    df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_most = sort_features(X=df_most, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "LSOSS"), end="\r")
    estimator = LSOSS(direction=direction, calculate_pval=False)
    df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_lsoss = sort_features(X=df_lsoss, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "DIDS"), end="\r")
    estimator = DIDS(score_function="quad", direction=direction, calculate_pval=False)
    df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_dids = sort_features(X=df_dids, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (zscore)"), end="\r")
    estimator = UHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_z = estimator.fit_predict(X=X, y=y)
    df_uhet_z = sort_features(X=df_uhet_z, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (robust)"), end="\r")
    estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_r = estimator.fit_predict(X=X, y=y)
    df_uhet_r = sort_features(X=df_uhet_r, features_name=features_name, X_map=False, map_genes=False)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "GLOBULIN"))
    estimator = GLOBULIN(normalize="robust", q=0.75, iqr_range=(25, 75), num_subsamples=50, subsampling_size=None,
                         significant_p=0.05, partition_by_anova=False, num_components=10, num_subclusters=10,
                         binary_clustering=True, calculate_pval=False, num_rounds=50, num_jobs=num_jobs)
    df_glob = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    df_glob = sort_features(X=df_glob, features_name=features_name, X_map=None, map_genes=False)
    current_progress += 1

    methods_df = [("COPA", df_copa), ("OS", df_os), ("ORT", df_ort), ("MOST", df_most),
                  ("LSOSS", df_lsoss), ("DIDS", df_dids), ("U-Het (zscore)", df_uhet_z),
                  ("U-Het (robust)", df_uhet_r), ("GLOBULIN", df_glob)]

    print("## Scoring results using known regulated features...")
    selected_regulated_features = top_k_features
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t\t>> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for stat_name, df in methods_df:
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        score = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(score)

    print("## Plot barplot using top k features...")
    plot_barplot(X=list_scores, methods_name=methods_name, file_name=file_name,
                 save_path=RESULT_PATH)

    print("## Plot results using top k features...")
    for idx, item in enumerate(methods_df):
        stat_name, df = item
        if total_progress == idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((idx + 1) / total_progress) * 100,
                                                                    stat_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((idx + 1) / total_progress) * 100,
                                                                    stat_name), end="\r")
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:top_k_features].tolist()]
        plot_umap(X=X[:, temp], y=y, num_features=top_k_features, standardize=True, num_jobs=num_jobs,
                  suptitle=stat_name.upper(), file_name=file_name + "_" + methods_name[idx].lower(),
                  save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
