import os

import numpy as np
import pandas as pd
import seaborn as sns

from model.copa import COPA
from model.deltaiqr import DeltaIQR
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.phet import PHeT
from model.studentt import StudentTTest
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_style(style='white')


def construct_data(X, y, features_name: list, regulated_features: list, control_class: int = 0, num_outliers: int = 5,
                   num_features_changes: int = 5, file_name: str = "synset",
                   save_path: str = "."):
    y = np.reshape(y, (y.shape[0], 1))
    regulated_features_idx = np.where(regulated_features != 0)[0]

    # Change feature expression for selected case samples
    X_temp = np.copy(X)
    picked_features = np.random.choice(a=len(regulated_features_idx),
                                       size=num_features_changes, replace=False)
    picked_features = regulated_features_idx[picked_features]
    for class_idx in np.unique(y):
        if class_idx == control_class:
            continue
        sample_idx = np.where(y == class_idx)[0]
        sigma = np.std(X_temp[sample_idx], axis=0)
        choice_idx = np.random.choice(a=sample_idx, size=num_outliers, replace=False)
        for idx in choice_idx:
            X_temp[idx, picked_features] += sigma[picked_features]
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_features.csv"), sep=",", index=False)
    df = pd.DataFrame(choice_idx, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_features_outliers.csv"), sep=",", index=False)
    df = pd.DataFrame(np.unique(picked_features), columns=["features"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_features_idx.csv"), sep=",", index=False)

    # Change feature expression for selected case and control samples
    X_temp = np.copy(X)
    temp_list = list()
    picked_features = np.random.choice(a=len(regulated_features_idx),
                                       size=num_features_changes, replace=False)
    picked_features = regulated_features_idx[picked_features]
    for class_idx in np.unique(y):
        sample_idx = np.where(y == class_idx)[0]
        sigma = np.std(X_temp[sample_idx], axis=0)
        choice_idx = np.random.choice(a=sample_idx, size=num_outliers, replace=False)
        temp_list.extend(choice_idx)
        for idx in choice_idx:
            X_temp[idx, picked_features] += sigma[picked_features]
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features.csv"), sep=",", index=False)
    df = pd.DataFrame(temp_list, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features_outliers.csv"), sep=",", index=False)
    df = pd.DataFrame(np.unique(picked_features), columns=["features"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features_idx.csv"), sep=",", index=False)


def train(num_jobs: int = 4):
    # Actions
    build_simulation = False

    # Arguments
    direction = "both"
    pvalue = 0.01
    num_features_changes = 46
    list_data = list(range(1, 11))
    data_type = ["minority_features", "mixed_features"]
    methods_name = ["ttest", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "DeltaIQR", "PHet"]

    # dataset name
    file_name = "simulated_normal"
    regulated_features_file = "simulated_normal_features.csv"

    # Load up/down regulated features
    true_regulated_features = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file), sep=',')
    true_regulated_features = true_regulated_features.to_numpy().squeeze()
    true_regulated_features[true_regulated_features < 0] = 1

    if build_simulation:
        # Load expression data
        X = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), sep=',')
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
        num_examples, num_features = X.shape
        for n_outliers in list_data:
            construct_data(X=X, y=y, features_name=features_name, regulated_features=true_regulated_features,
                           control_class=0, num_outliers=n_outliers, num_features_changes=num_features_changes,
                           file_name=file_name + "%.2d" % n_outliers, save_path=DATASET_PATH)

    total_progress = len(methods_name)
    methods_features = np.zeros(shape=(len(methods_name), len(list_data) * len(data_type)), dtype=np.int32)
    methods_outliers_scores = np.zeros(shape=(len(methods_name), len(list_data) * len(data_type)), dtype=np.float32)
    for data_idx, outlier_idx in enumerate(list_data):
        for type_idx, outlier_type in enumerate(data_type):
            current_progress = 1
            temp_name = file_name + "%.2d_%s" % (outlier_idx, outlier_type)
            print("## Perform simulation studies using {0} data...".format(temp_name))
            # Load expression data
            X = pd.read_csv(os.path.join(DATASET_PATH, temp_name + ".csv"), sep=',')
            y = X["class"].to_numpy()
            features_name = X.drop(["class"], axis=1).columns.to_list()
            X = X.drop(["class"], axis=1).to_numpy()
            num_examples, num_features = X.shape
            # Load outliers
            X_outlier = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_outliers.csv"), sep=',')
            X_outlier = np.array(X_outlier["samples"])
            temp = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_idx.csv"), sep=',')
            temp = np.array(temp["features"])
            true_changed_features = np.zeros((num_features))
            true_changed_features[temp] = 1

            current_idx = (data_idx * len(data_type)) + type_idx
            print("\t>> Sample size: {0}; Feature size: {1}; Class size: {2}".format(num_examples, num_features,
                                                                                     len(np.unique(y))))

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[0]), end="\r")
            estimator = StudentTTest(direction=direction, calculate_pval=False)
            df_ttest = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[1]), end="\r")
            estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
            df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[2]), end="\r")
            estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                            calculate_pval=False)
            df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[3]), end="\r")
            estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                               calculate_pval=False)
            df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[4]), end="\r")
            estimator = MOST(direction=direction, calculate_pval=False)
            df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[5]), end="\r")
            estimator = LSOSS(direction=direction, calculate_pval=False)
            df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[6]), end="\r")
            estimator = DIDS(score_function="tanh", direction=direction, calculate_pval=False)
            df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[7]), end="\r")
            temp = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_deco.csv"), sep=',')
            df_deco = pd.DataFrame([(features_name[int(item[1][0])], item[1][1])
                                    for item in temp.iterrows()], columns=["features", "score"])
            del temp
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[8]), end="\r")
            estimator = DeltaIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
            df_iqr = estimator.fit_predict(X=X, y=y)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[9]))
            estimator = PHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=1000, subsampling_size=None,
                             significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                             weight_range=[0.1, 0.4, 0.8], calculate_hstatistic=False, num_components=10,
                             num_subclusters=10, binary_clustering=True, calculate_pval=False, num_rounds=50,
                             num_jobs=num_jobs)
            df_phet = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            methods_dict = dict({methods_name[0]: df_ttest, methods_name[1]: df_copa, methods_name[2]: df_os,
                                 methods_name[3]: df_ort, methods_name[4]: df_most, methods_name[5]: df_lsoss,
                                 methods_name[6]: df_dids, methods_name[7]: df_deco, methods_name[8]: df_iqr,
                                 methods_name[9]: df_phet})

            print("\t>> Sort features by the score statistic...".format())
            for method_idx, item in enumerate(methods_dict.items()):
                method_name, df = item
                if methods_name[method_idx] == "DECO":
                    temp = [idx for idx, feature in enumerate(features_name)
                            if feature in df['features'].tolist()]
                else:
                    temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                                X_map=None, map_genes=False, ttest=False)
                    temp = [idx for idx, feature in enumerate(features_name)
                            if feature in temp['features'].tolist()]
                methods_features[method_idx, current_idx] = len(temp)
                temp = sort_features(X=df, features_name=features_name, X_map=None,
                                     map_genes=False, ttest=False)
                methods_dict[method_name] = temp
            del df_copa, df_os, df_ort, df_most, df_lsoss, df_dids, df_deco, df_iqr, df_phet

            print("\t>> Scoring results using known regulated features and outliers...")
            for method_idx, item in enumerate(methods_dict.items()):
                if method_idx + 1 == len(methods_name):
                    print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(
                        ((method_idx + 1) / len(methods_name)) * 100,
                        methods_name[method_idx]))
                else:
                    print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(methods_name)) * 100,
                                                                              methods_name[method_idx]), end="\r")
                method_name, df = item
                temp = [idx for idx, feature in enumerate(features_name)
                        if feature in df['features'].tolist()[:num_features_changes]]
                pred_changed_features = np.zeros((num_features))
                pred_changed_features[temp] = 1
                score = comparative_score(pred_features=pred_changed_features,
                                          true_features=true_changed_features,
                                          metric="f1")
                methods_outliers_scores[method_idx, current_idx] = score

    temp = [outlier_type + "%.2d" % outlier_idx for outlier_idx in list_data for outlier_type in data_type]
    df = pd.DataFrame(methods_features, columns=temp, index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_features.csv"), sep=",")
    df = pd.DataFrame(methods_outliers_scores, columns=temp, index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_outliers_scores.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)

# Outliers
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated_normal_methods_outliers_scores.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()
methods_name = df.index.to_list()
methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
df.index = methods_name

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case samples)", fontsize=22)
ax.set_ylabel("F1 scores of each method", fontsize=22)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=18)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("F1 scores of each method", fontsize=22)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=18)

# Features
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated_normal_methods_features.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()
methods_name = df.index.to_list()
methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
df.index = methods_name

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case samples)", fontsize=22)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=22)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=18)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=22)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=18)

# # Legend
# ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10))
# ax.set_xlabel("Number of outliers (in case and control samples)")
# ax.set_ylabel("Number of significant features found by each method")
# ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
# ax.legend(title="Methods", title_fontsize=30, fontsize=25, ncol=5,
#           loc="lower right", bbox_to_anchor=(1.0, 1.0),
#           facecolor="None")