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
from model.tstatistic import StudentTTest
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_style(style='white')


def construct_data(X, y, features_name: list, regulated_features: list, control_class: int = 0,
                   num_outliers: int = 5, num_features_changes: int = 5, file_name: str = "synset",
                   save_path: str = "."):
    y = np.reshape(y, (y.shape[0], 1))
    regulated_features_idx = np.where(regulated_features != 0)[0]
    if num_features_changes > len(regulated_features_idx):
        temp = [idx for idx in range(X.shape[1]) if idx not in regulated_features_idx]
        temp = np.random.choice(a=temp, size=num_features_changes - len(regulated_features_idx),
                                replace=False)
        regulated_features_idx = np.append(regulated_features_idx, temp)

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
            X_temp[idx, picked_features] += sigma[picked_features] * 1.5
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
            X_temp[idx, picked_features] += sigma[picked_features] * 1.5
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features.csv"), sep=",", index=False)
    df = pd.DataFrame(temp_list, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features_outliers.csv"), sep=",", index=False)
    df = pd.DataFrame(np.unique(picked_features), columns=["features"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features_idx.csv"), sep=",", index=False)


def train():
    # Actions
    build_simulation = False

    # Arguments
    direction = "both"
    pvalue = 0.01
    num_features_changes = 100
    list_data = list(range(1, 11))
    data_type = ["minority_features", "mixed_features"]
    methods = ["t-statistic", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "ΔIQR", "PHet"]

    # dataset name
    file_name = "simulated_normal"
    regulated_features_file = "simulated_normal_features.csv"

    # Load up/down regulated features
    regulated_features = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file), sep=',')
    regulated_features = regulated_features.to_numpy().squeeze()
    regulated_features[regulated_features < 0] = 1

    if build_simulation:
        # Load expression data
        X = pd.read_csv(os.path.join(
            DATASET_PATH, file_name + ".csv"), sep=',')
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
        num_examples, num_features = X.shape
        for n_outliers in list_data:
            construct_data(X=X, y=y, features_name=features_name, regulated_features=regulated_features,
                           control_class=0, num_outliers=n_outliers, num_features_changes=num_features_changes,
                           file_name=file_name + "%.2d" % n_outliers, save_path=DATASET_PATH)

    total_progress = len(methods)
    methods_features = np.zeros(shape=(len(methods), len(list_data) * len(data_type)), dtype=np.int32)
    methods_outliers_scores = np.zeros(shape=(len(methods), len(list_data) * len(data_type)), dtype=np.float32)
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
            temp_sign = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_idx.csv"), sep=',')
            temp_sign = np.array(temp_sign["features"])
            true_changed_features = np.zeros((num_features))
            true_changed_features[temp_sign] = 1

            current_idx = (data_idx * len(data_type)) + type_idx
            print("\t>> Sample size: {0}; Feature size: {1}; Class size: {2}".format(num_examples, num_features,
                                                                                     len(np.unique(y))))

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[0]), end="\r")
            estimator = StudentTTest(direction=direction, permutation_test=False)
            df_ttest = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[1]), end="\r")
            estimator = COPA(q=0.75, direction=direction, permutation_test=False)
            df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[2]), end="\r")
            estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                            permutation_test=False)
            df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[3]), end="\r")
            estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                               permutation_test=False)
            df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[4]), end="\r")
            estimator = MOST(direction=direction, permutation_test=False)
            df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[5]), end="\r")
            estimator = LSOSS(direction=direction, permutation_test=False)
            df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[6]), end="\r")
            estimator = DIDS(score_function="tanh", direction=direction, permutation_test=False)
            df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[7]), end="\r")
            temp_sign = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_deco.csv"), sep=',')
            df_deco = pd.DataFrame([(features_name[int(item[1][0])], item[1][1])
                                    for item in temp_sign.iterrows()], columns=["features", "score"])
            del temp_sign
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[8]), end="\r")
            estimator = DeltaIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), permutation_test=False)
            df_iqr = estimator.fit_predict(X=X, y=y)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods[9]))
            estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, calculate_deltaiqr=True,
                             calculate_fisher=True, calculate_profile=True, bin_KS_pvalues=True,
                             feature_weight=[0.4, 0.3, 0.2, 0.1], weight_range=[0.1, 0.4, 0.8])
            df_phet = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            methods_dict = dict({methods[0]: df_ttest, methods[1]: df_copa, methods[2]: df_os,
                                 methods[3]: df_ort, methods[4]: df_most, methods[5]: df_lsoss,
                                 methods[6]: df_dids, methods[7]: df_deco, methods[8]: df_iqr,
                                 methods[9]: df_phet})
            del df_copa, df_os, df_ort, df_most, df_lsoss, df_dids, df_deco, df_iqr, df_phet
            print("\t>> Scoring results using known regulated features and outliers...")
            for method_idx, item in enumerate(methods_dict.items()):
                method_name, df = item
                if method_idx + 1 == len(methods):
                    print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(
                        ((method_idx + 1) / len(methods)) * 100, method_name))
                else:
                    print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(methods)) * 100,
                                                                              method_name), end="\r")
                if method_name == "DECO":
                    temp_sign = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
                else:
                    temp_sign = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                                     X_map=None, map_genes=False, ttest=False)
                    temp_sign = [idx for idx, feature in enumerate(features_name)
                                 if feature in temp_sign['features'].tolist()]
                    temp_sort = sort_features(X=df, features_name=features_name, X_map=None,
                                              map_genes=False, ttest=False)
                    temp_sort = [features_name.index(item) for item in temp_sort['features'].tolist()]
                methods_features[method_idx, current_idx] = len(temp_sign)
                temp_sort = temp_sort[:num_features_changes]
                pred_changed_features = np.zeros((num_features))
                pred_changed_features[temp_sort] = 1
                score = comparative_score(pred_features=pred_changed_features,
                                          true_features=true_changed_features,
                                          metric="f1")
                methods_outliers_scores[method_idx, current_idx] = score

    temp = [outlier_type + "%.2d" % outlier_idx for outlier_idx in list_data for outlier_type in data_type]
    df = pd.DataFrame(methods_features, columns=temp, index=methods)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_features.csv"), sep=",")
    df = pd.DataFrame(methods_outliers_scores, columns=temp, index=methods)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_outliers_scores.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train()
