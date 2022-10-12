import os

import numpy as np
import pandas as pd
import seaborn as sns

from model.cleanse import CLEANSE
from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import outliers_analysis
from utility.utils import sort_features, significant_features

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
    direction = "both"
    pvalue = 0.01
    calculate_hstatistic = False
    sort_by_pvalue = True
    topKfeatures = 100
    plot_topKfeatures = False
    if not sort_by_pvalue:
        plot_topKfeatures = True

    # Datasets:
    # 1. simulated_normal, simulated_normal_minority, simulated_normal_minority_features, 
    # simulated_normal_mixed, simulated_normal_mixed_features
    # 2. simulated_weak, simulated_weak_minority, simulated_weak_minority_features, 
    # simulated_weak_mixed, simulated_weak_mixed_features
    file_name = "simulated_weak_mixed_features"
    regulated_features_file = "simulated_weak_features.csv"

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
    print(
        "\t >> Sample size: {0}; Feature size: {1}; Class size: {2}".format(X.shape[0], X.shape[1], len(np.unique(y))))
    current_progress = 1
    total_progress = 11

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "COPA"), end="\r")
    estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
    df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "OS"), end="\r")
    estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                    calculate_pval=False)
    df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "ORT"), end="\r")
    estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                       calculate_pval=False)
    df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "MOST"), end="\r")
    estimator = MOST(direction=direction, calculate_pval=False)
    df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "LSOSS"), end="\r")
    estimator = LSOSS(direction=direction, calculate_pval=False)
    df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "DIDS"), end="\r")
    estimator = DIDS(score_function="quad", direction=direction, calculate_pval=False)
    df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "DECO"), end="\r")
    temp = pd.read_csv(os.path.join(DATASET_PATH, file_name + "_deco.csv"), sep=',')
    df_deco = pd.DataFrame([(features_name[int(item[1][0])], item[1][1])
                            for item in temp.iterrows()], columns=["features", "score"])
    del temp
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (zscore)"), end="\r")
    estimator = UHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_z = estimator.fit_predict(X=X, y=y)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (robust)"), end="\r")
    estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_r = estimator.fit_predict(X=X, y=y)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "CLEANSE (zscore)"), end="\r")
    estimator = CLEANSE(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=1000, subsampling_size=None,
                        significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                        calculate_hstatistic=calculate_hstatistic, num_components=10, num_subclusters=10,
                        binary_clustering=True, calculate_pval=False, num_rounds=50, num_jobs=num_jobs)
    df_cleanse_z = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "CLEANSE (robust)"))
    estimator = CLEANSE(normalize="robust", q=0.75, iqr_range=(25, 75), num_subsamples=1000, subsampling_size=None,
                        significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                        calculate_hstatistic=calculate_hstatistic, num_components=10, num_subclusters=10,
                        binary_clustering=True, calculate_pval=False, num_rounds=50, num_jobs=num_jobs)
    df_cleanse_r = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)

    methods_df = dict({"COPA": df_copa, "OS": df_os, "ORT": df_ort, "MOST": df_most, "LSOSS": df_lsoss,
                       "DIDS": df_dids, "DECO": df_deco, "UHet_zscore": df_uhet_z, "UHet_robust": df_uhet_r,
                       "CLEANSE_zscore": df_cleanse_z, "CLEANSE_robust": df_cleanse_r})

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
    else:
        print("## Sort features by the score statistic...".format())
    for stat_name, df in methods_df.items():
        if stat_name == "DECO":
            continue
        if sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                            X_map=None, map_genes=False, ttest=False)
        else:
            temp = sort_features(X=df, features_name=features_name, X_map=None,
                                 map_genes=False, ttest=False)
        methods_df[stat_name] = temp
    del df_copa, df_os, df_ort, df_most, df_lsoss, df_dids, df_uhet_z, df_uhet_r, df_cleanse_r

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

    if plot_topKfeatures:
        print("## Plot results using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot results using the top features for each method...")
    for method_idx, item in enumerate(methods_df.items()):
        stat_name, df = item
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    stat_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    stat_name), end="\r")
        if plot_topKfeatures:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:topKfeatures]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        plot_umap(X=X[:, temp], y=y, num_features=num_features, standardize=True, num_jobs=num_jobs,
                  suptitle=stat_name.upper(), file_name=file_name + "_" + stat_name.lower(),
                  save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)
