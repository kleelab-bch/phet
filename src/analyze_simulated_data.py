import os

import numpy as np
import pandas as pd
import seaborn as sns

from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import outliers_analysis
from utility.utils import sort_features, comparative_score

sns.set_theme()


def construct_data(X, y, features_name: list, regulated_features: list, control_class: int = 0, num_outliers: int = 5,
                   variance: float = 1.5, file_name: str = "synset", save_path: str = "."):
    y = np.reshape(y, (y.shape[0], 1))
    regulated_features_idx = np.where(regulated_features != 0)[0]

    # Minority change wrt to case samples
    X_temp = np.copy(X)
    for class_idx in np.unique(y):
        if class_idx == control_class:
            continue
        case_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=case_idx, size=num_outliers, replace=False)
        X_temp[choice_idx] = X_temp[choice_idx] * variance
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority.csv"), sep=",", index=False)

    # Mixed change wrt to control and case samples
    X_temp = np.copy(X)
    for class_idx in np.unique(y):
        sample_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=sample_idx, size=num_outliers, replace=False)
        X_temp[choice_idx] = X_temp[choice_idx] * variance
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed.csv"), sep=",", index=False)

    # Exchange feature expression wrt to case samples
    control_idx = np.where(y == control_class)[0]
    X_control = X[control_idx][:, regulated_features_idx]
    mu = np.mean(X_control, axis=0)
    sigma = np.std(X_control, axis=0)
    X_temp = np.copy(X)
    for class_idx in np.unique(y):
        if class_idx == control_class:
            continue
        case_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=case_idx, size=num_outliers, replace=False)
        for idx in choice_idx:
            picked_features = np.random.choice(a=len(regulated_features_idx),
                                               size=num_outliers, replace=False)
            temp = regulated_features_idx[picked_features]
            X_temp[idx, temp] = np.random.normal(loc=mu[picked_features],
                                                 scale=sigma[picked_features])
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_features.csv"), sep=",", index=False)

    # Exchange feature expression wrt to control and case samples
    X_temp = np.copy(X)
    for class_idx in np.unique(y):
        control_idx = np.random.choice(a=[idx for idx in np.unique(y) if idx != class_idx],
                                       size=1)
        X_control = X[control_idx][:, regulated_features_idx]
        mu = np.mean(X_control, axis=0)
        sigma = np.std(X_control, axis=0)

        case_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(
            a=case_idx, size=num_outliers, replace=False)
        for idx in choice_idx:
            picked_features = np.random.choice(a=len(regulated_features_idx),
                                               size=num_outliers, replace=False)
            temp = regulated_features_idx[picked_features]
            X_temp[idx, temp] = np.random.normal(loc=mu[picked_features],
                                                 scale=sigma[picked_features])
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features.csv"), sep=",", index=False)


def train(num_jobs: int = 4):
    # Actions
    analyze_outliers = False
    build_simulation = False

    # Arguments
    top_k_features = 100
    direction = "both"

    # Datasets:
    # 1. simulated_normal, simulated_normal_minority, simulated_normal_minority_features, 
    # simulated_normal_mixed, simulated_normal_mixed_features
    # 2. simulated_weak, simulated_weak_minority, simulated_weak_minority_features, 
    # simulated_weak_mixed, simulated_weak_mixed_features
    regulated_features_file = "simulated_normal_features.csv"
    file_name = "simulated_normal"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), sep=',')
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file), sep=',')
    top_features_true = top_features_true.to_numpy().squeeze()
    top_features_true[top_features_true < 0] = 1
    if analyze_outliers:
        print("## Analyzing outliers...")
        outliers_analysis(X=X, y=y, regulated_features=top_features_true)
    if build_simulation:
        print("## Constructing four simulated data...")
        construct_data(X=X, y=y, features_name=features_name, regulated_features=top_features_true,
                       control_class=0, num_outliers=5, variance=1.5, file_name=file_name,
                       save_path=DATASET_PATH)

    print("## Perform simulation studies using {0} data...".format(file_name))
    methods_name = ["COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "UHet_zscore", "UHet_robust"]
    current_progress = 1
    total_progress = 8

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
                                                            "U-Het (robust)"))
    estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_r = estimator.fit_predict(X=X, y=y)
    df_uhet_r = sort_features(X=df_uhet_r, features_name=features_name, X_map=False, map_genes=False)

    methods_df = [("COPA", df_copa), ("OS", df_os), ("ORT", df_ort), ("MOST", df_most),
                  ("LSOSS", df_lsoss), ("DIDS", df_dids), ("U-Het (zscore)", df_uhet_z),
                  ("U-Het (robust)", df_uhet_r)]

    print("## Scoring results using known regulated features...")
    selected_regulated_features = top_k_features
    temp = top_features_true.sum()
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
        if len(methods_name) == idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((idx + 1) / len(methods_name)) * 100,
                                                                    stat_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((idx + 1) / len(methods_name)) * 100,
                                                                    stat_name), end="\r")
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:top_k_features].tolist()]
        plot_umap(X=X[:, temp], y=y, num_features=top_k_features, standardize=True, num_jobs=num_jobs,
                  suptitle=stat_name.upper(), file_name=file_name + "_" + methods_name[idx].lower(),
                  save_path=RESULT_PATH)
        # plot_clusters(X=X[:, temp], y=y, features_name=features_name[:top_k_features], num_features=top_k_features,
        #               standardize=True, cluster_type="spectral", num_clusters=0, num_neighbors=15, min_dist=0,
        #               heatmap=True, proportion=True, show_umap=True, num_jobs=num_jobs, suptitle=stat_name.upper(),
        #               file_name=stat_name.lower(), save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
