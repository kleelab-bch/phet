import os
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns

from model.copa import COPA
from model.deltahvfmean import DeltaHVFMean
from model.deltaiqrmean import DeltaIQRMean
from model.dids import DIDS
from model.hvf import SeuratHVF, HIQR
from model.lsoss import LSOSS
from model.most import MOST
from model.nonparametric_test import StudentTTest, WilcoxonRankSumTest, KolmogorovSmirnovTest
from model.ort import OutlierRobustTstatistic
from model.oss import OutlierSumStatistic
from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma",
           "KS", "KS+Gamma", "LIMMA", "LIMMA+Gamma", "Dispersion (composite)",
           "Dispersion (by condition)", "ΔDispersion", "ΔDispersion+ΔMean",
           "IQR (composite)", "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean",
           "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO",
           "PHet (ΔDispersion)", "PHet"]


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

    # Filtering arguments
    alpha = 0.01
    num_features_changes = 100

    # Models parameters
    direction = "both"

    # dataset name
    list_data = list(range(1, 11))
    data_type = ["minority_features", "mixed_features"]
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

    total_progress = len(METHODS)
    methods_features = np.zeros(shape=(len(METHODS), len(list_data) * len(data_type)), dtype=np.int32)
    methods_outliers_scores = np.zeros(shape=(len(METHODS), len(list_data) * len(data_type)), dtype=np.float32)
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

            current_progress = 1
            total_progress = len(METHODS)
            methods_dict = dict()

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[0]), end="\r")
            estimator = StudentTTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                     adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False, ttest=False,
                               ascending=True)
            df = df[df["score"] < alpha]
            methods_dict.update({METHODS[0]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[1]), end="\r")
            estimator = StudentTTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                     adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[1]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[2]), end="\r")
            estimator = WilcoxonRankSumTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                            adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False, ttest=False,
                               ascending=True)
            df = df[df["score"] < alpha]
            methods_dict.update({METHODS[2]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[3]), end="\r")
            estimator = WilcoxonRankSumTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                            adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[3]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[4]), end="\r")
            estimator = KolmogorovSmirnovTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                              adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False, ttest=False,
                               ascending=True)
            df = df[df["score"] < alpha]
            methods_dict.update({METHODS[4]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[5]), end="\r")
            estimator = KolmogorovSmirnovTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                              adjusted_alpha=alpha)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[5]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[6]), end="\r")
            df = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_limma_features.csv"), sep=',')
            df = df[["ID", "adj.P.Val", "B"]]
            df = df[df["adj.P.Val"] < alpha]
            df = df[["ID", "B"]]
            df["ID"] = list(np.array(features_name)[df["ID"].to_list()])
            df.columns = ["features", "score"]
            methods_dict.update({METHODS[6]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[7]), end="\r")
            df = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_limma_features.csv"), sep=',')
            df = df[["ID", "B"]]
            df["ID"] = list(np.array(features_name)[df["ID"].to_list()])
            temp = [df["ID"].to_list().index(item) for item in features_name]
            df = np.absolute(df.iloc[temp]["B"].to_numpy()[:, None])
            methods_dict.update({METHODS[7]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[8]), end="\r")
            estimator = SeuratHVF(per_condition=False, log_transform=True, num_top_features=num_features,
                                  min_disp=0.5, min_mean=0.0125, max_mean=3)
            temp_X = deepcopy(X)
            df = estimator.fit_predict(X=temp_X, y=y)
            del temp_X
            methods_dict.update({METHODS[8]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[9]), end="\r")
            estimator = SeuratHVF(per_condition=True, log_transform=True, num_top_features=num_features,
                                  min_disp=0.5, min_mean=0.0125, max_mean=3)
            temp_X = deepcopy(X)
            df = estimator.fit_predict(X=temp_X, y=y)
            del temp_X
            methods_dict.update({METHODS[9]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[10]), end="\r")
            estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=True, num_top_features=num_features,
                                     min_disp=0.5, min_mean=0.0125, max_mean=3)
            temp_X = deepcopy(X)
            df = estimator.fit_predict(X=temp_X, y=y)
            del temp_X
            methods_dict.update({METHODS[10]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[11]), end="\r")
            estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=True, num_top_features=num_features,
                                     min_disp=0.5, min_mean=0.0125, max_mean=3)
            temp_X = deepcopy(X)
            df = estimator.fit_predict(X=temp_X, y=y)
            del temp_X
            methods_dict.update({METHODS[11]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[12]), end="\r")
            estimator = HIQR(per_condition=False, normalize="zscore", iqr_range=(25, 75))
            df = estimator.fit_predict(X=X, y=y)
            methods_dict.update({METHODS[12]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[13]), end="\r")
            estimator = HIQR(per_condition=True, normalize="zscore", iqr_range=(25, 75))
            df = estimator.fit_predict(X=X, y=y)
            methods_dict.update({METHODS[13]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[14]), end="\r")
            estimator = DeltaIQRMean(calculate_deltamean=False, normalize="zscore", iqr_range=(25, 75))
            df = estimator.fit_predict(X=X, y=y)
            methods_dict.update({METHODS[14]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[15]), end="\r")
            estimator = DeltaIQRMean(calculate_deltamean=True, normalize="zscore", iqr_range=(25, 75))
            df = estimator.fit_predict(X=X, y=y)
            methods_dict.update({METHODS[15]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[16]), end="\r")
            estimator = COPA(q=75)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[16]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[17]), end="\r")
            estimator = OutlierSumStatistic(q=75, iqr_range=(25, 75), two_sided_test=False)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[17]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[18]), end="\r")
            estimator = OutlierRobustTstatistic(q=75, iqr_range=(25, 75))
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[18]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[19]), end="\r")
            estimator = MOST()
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[19]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[20]), end="\r")
            estimator = LSOSS(direction=direction)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[20]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[21]), end="\r")
            estimator = DIDS(score_function="tanh", direction=direction)
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[21]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      METHODS[22]), end="\r")
            df = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_deco_features.csv"), sep=',')
            df = pd.DataFrame([(features_name[int(item[1][0])], item[1][1]) for item in df.iterrows()],
                              columns=["features", "score"])
            methods_dict.update({METHODS[22]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[23]), end="\r")
            estimator = PHeT(normalize="log", iqr_range=(25, 75), num_subsamples=1000, delta_type="hvf",
                             calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                             calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                             weight_range=[0.2, 0.4, 0.8])
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[23]: df})
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                      METHODS[24]))
            estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                             calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                             calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                             weight_range=[0.2, 0.4, 0.8])
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            methods_dict.update({METHODS[24]: df})

            print("\t>> Scoring results using known regulated features and outliers...")
            for method_idx, item in enumerate(methods_dict.items()):
                method_name, df = item
                if method_idx + 1 == len(METHODS):
                    print(
                        "\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / len(METHODS)) * 100,
                                                                            method_name))
                else:
                    print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(METHODS)) * 100,
                                                                              method_name), end="\r")
                if method_name in ['DECO', 't-statistic', 'Wilcoxon', 'LIMMA', 'KS']:
                    temp_sign = [features_name.index(item) for item in df['features'].tolist()]
                    temp_sort = temp_sign
                else:
                    temp_sign = significant_features(X=df, features_name=features_name, alpha=alpha,
                                                     X_map=None, map_genes=False, ttest=False)
                    temp_sign = [idx for idx, feature in enumerate(features_name)
                                 if feature in temp_sign['features'].tolist()]
                    temp_sort = sort_features(X=df, features_name=features_name, X_map=None,
                                              map_genes=False, ttest=False)
                    temp_sort = [features_name.index(item) for item in temp_sort['features'].tolist()]
                # Store the number of predicted important features by each method
                methods_features[method_idx, current_idx] = len(temp_sign)
                temp_sort = temp_sort[:num_features_changes]
                pred_changed_features = np.zeros((num_features))
                pred_changed_features[temp_sort] = 1
                score = comparative_score(pred_features=pred_changed_features,
                                          true_features=true_changed_features,
                                          metric="f1")
                # Compare the 100 top true feature with the top 100 predicted 
                # important features by each method
                methods_outliers_scores[method_idx, current_idx] = score

    temp = [outlier_type + "%.2d" % outlier_idx for outlier_idx in list_data for outlier_type in data_type]
    df = pd.DataFrame(methods_features, columns=temp, index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_features.csv"), sep=",")
    df = pd.DataFrame(methods_outliers_scores, columns=temp, index=METHODS)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_outliers_scores.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train()
