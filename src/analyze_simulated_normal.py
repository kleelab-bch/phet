import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import Counter

from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.phet import PHeT
from model.uhet import DiffIQR
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import dimensionality_reduction, clustering, comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()


def construct_data(X, y, features_name: list, regulated_features: list, control_class: int = 0, num_outliers: int = 5,
                   variance: float = 1.5, file_name: str = "synset", save_path: str = "."):
    y = np.reshape(y, (y.shape[0], 1))
    regulated_features_idx = np.where(regulated_features != 0)[0]

    # Minority change w.r.t. to case samples
    X_temp = np.copy(X)
    for class_idx in np.unique(y):
        if class_idx == control_class:
            continue
        case_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=case_idx, size=num_outliers, replace=False)
        X_temp[choice_idx] = X_temp[choice_idx] * variance
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority.csv"), sep=",", index=False)
    df = pd.DataFrame(choice_idx, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_idx.csv"), sep=",", index=False)

    # Mixed change w.r.t. to control and case samples
    X_temp = np.copy(X)
    temp_list = list()
    for class_idx in np.unique(y):
        sample_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=sample_idx, size=num_outliers, replace=False)
        X_temp[choice_idx] = X_temp[choice_idx] * variance
        temp_list.extend(choice_idx)
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed.csv"), sep=",", index=False)
    df = pd.DataFrame(temp_list, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_idx.csv"), sep=",", index=False)

    # Exchange feature expression w.r.t. to case samples
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
    df = pd.DataFrame(choice_idx, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_minority_features_idx.csv"), sep=",", index=False)

    # Exchange feature expression w.r.t. to control and case samples
    X_temp = np.copy(X)
    temp_list = list()
    for class_idx in np.unique(y):
        control_idx = np.random.choice(a=[idx for idx in np.unique(y) if idx != class_idx],
                                       size=1)
        X_control = X[control_idx][:, regulated_features_idx]
        mu = np.mean(X_control, axis=0)
        sigma = np.std(X_control, axis=0)

        case_idx = np.where(y == class_idx)[0]
        choice_idx = np.random.choice(a=case_idx, size=num_outliers, replace=False)
        temp_list.extend(choice_idx)
        for idx in choice_idx:
            picked_features = np.random.choice(a=len(regulated_features_idx),
                                               size=num_outliers, replace=False)
            temp = regulated_features_idx[picked_features]
            X_temp[idx, temp] = np.random.normal(loc=mu[picked_features],
                                                 scale=sigma[picked_features])
    df = pd.DataFrame(np.hstack((y, X_temp)), columns=["class"] + features_name)
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features.csv"), sep=",", index=False)
    df = pd.DataFrame(temp_list, columns=["samples"])
    df.to_csv(path_or_buf=os.path.join(save_path, file_name + "_mixed_features_idx.csv"), sep=",", index=False)


def train(num_jobs: int = 4):
    # Actions
    build_simulation = False

    # Arguments
    direction = "both"
    pvalue = 0.01
    calculate_hstatistic = False
    sort_by_pvalue = True
    list_data = list(range(1, 11))
    # data_type = ["minority", "mixed", "minority_features", "mixed_features"]
    data_type = ["minority", "mixed"]
    methods_name = ["COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "DiffIQR", "PHeT"]

    # dataset name
    file_name = "simulated_normal"
    regulated_features_file = "simulated_normal_features.csv"

    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file), sep=',')
    top_features_true = top_features_true.to_numpy().squeeze()
    top_features_true[top_features_true < 0] = 1
    if build_simulation:
        # Load expression data
        X = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), sep=',')
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
        num_examples, num_features = X.shape
        for n_outliers in list_data:
            construct_data(X=X, y=y, features_name=features_name, regulated_features=top_features_true,
                           control_class=0, num_outliers=n_outliers, variance=1.5,
                           file_name=file_name + "%.2d" % n_outliers,
                           save_path=DATASET_PATH)

    total_progress = len(methods_name)
    methods_features = np.zeros(shape=(len(methods_name), len(list_data) * len(data_type)), dtype=np.int32)
    methods_features_scores = np.zeros(shape=(len(methods_name), len(list_data) * len(data_type)), dtype=np.float32)
    methods_outliers = np.zeros(shape=(len(methods_name), len(list_data) * len(data_type)), dtype=np.int32)
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
            current_idx = (data_idx * len(data_type)) + type_idx

            print("\t>> Sample size: {0}; Feature size: {1}; Class size: {2}".format(num_examples, num_features,
                                                                                     len(np.unique(y))))
            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[0]), end="\r")
            estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
            df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[1]), end="\r")
            estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                            calculate_pval=False)
            df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[2]), end="\r")
            estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                               calculate_pval=False)
            df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[3]), end="\r")
            estimator = MOST(direction=direction, calculate_pval=False)
            df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[4]), end="\r")
            estimator = LSOSS(direction=direction, calculate_pval=False)
            df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[5]), end="\r")
            estimator = DIDS(score_function="quad", direction=direction, calculate_pval=False)
            df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[6]), end="\r")
            temp = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_deco.csv"), sep=',')
            df_deco = pd.DataFrame([(features_name[int(item[1][0])], item[1][1])
                                    for item in temp.iterrows()], columns=["features", "score"])
            del temp
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[7]), end="\r")
            estimator = DiffIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
            df_uhet = estimator.fit_predict(X=X, y=y)
            current_progress += 1

            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                      methods_name[8]))
            estimator = PHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=5000, subsampling_size=None,
                             significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                             weight_range=[0.1, 0.4, 0.8], calculate_hstatistic=calculate_hstatistic, num_components=10,
                             num_subclusters=10, binary_clustering=True, calculate_pval=False, num_rounds=50,
                             num_jobs=num_jobs)
            df_phet = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
            current_progress += 1

            methods_dict = dict({methods_name[0]: df_copa, methods_name[1]: df_os, methods_name[2]: df_ort,
                                 methods_name[3]: df_most, methods_name[4]: df_lsoss, methods_name[5]: df_dids,
                                 methods_name[6]: df_deco, methods_name[7]: df_uhet, methods_name[8]: df_phet})
            if sort_by_pvalue:
                print("\t>> Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
            else:
                print("\t>> Sort features by the score statistic...".format())
            for method_idx, item in enumerate(methods_dict.items()):
                method_name, df = item
                if methods_name[method_idx] == "DECO":
                    continue
                if sort_by_pvalue:
                    temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                                X_map=None, map_genes=False, ttest=False)
                else:
                    temp = sort_features(X=df, features_name=features_name, X_map=None,
                                         map_genes=False, ttest=False)
                methods_dict[method_name] = temp
            del df_copa, df_os, df_ort, df_most, df_lsoss, df_dids, df_deco, df_uhet, df_phet

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
                temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
                top_features_pred = np.zeros((len(top_features_true)))
                top_features_pred[temp] = 1
                methods_features[method_idx, current_idx] = len(temp)
                methods_features_scores[method_idx, current_idx] = comparative_score(
                    top_features_pred=top_features_pred,
                    top_features_true=top_features_true,
                    metric="f1")
                # reduce data
                X_reducer = dimensionality_reduction(X=X[:, temp], num_neighbors=5, num_components=2, min_dist=0.0,
                                                     reduction_method="umap", num_epochs=2000, num_jobs=num_jobs)
                num_clusters = 3
                if type_idx == 1 or type_idx == 3:
                    num_clusters = 4
                cluster_labels = clustering(X=X_reducer, cluster_type="kmeans", num_clusters=num_clusters,
                                            num_jobs=num_jobs, predict=True)
                del X_reducer
                # Check outliers
                X_outlier = pd.read_csv(os.path.join(DATASET_PATH, temp_name + "_idx.csv"), sep=',')
                X_outlier = np.array(X_outlier["samples"])
                max_outlier_idx = Counter(cluster_labels[X_outlier]).most_common(1)
                methods_outliers[method_idx, current_idx] = max_outlier_idx[0][1]
                methods_outliers_scores[method_idx, current_idx] = max_outlier_idx[0][1] / sum(
                    cluster_labels == max_outlier_idx[0][0])
                del cluster_labels

    temp = [outlier_type + "%.2d" % outlier_idx for outlier_idx in list_data for outlier_type in data_type]
    df = pd.DataFrame(methods_features, columns=temp, index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_features.csv"), sep=",")
    df = pd.DataFrame(methods_features_scores, columns=temp, index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_features_scores.csv"), sep=",")
    df = pd.DataFrame(methods_outliers, columns=temp, index=methods_name)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_methods_outliers.csv"), sep=",")
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
methods_name = ["ΔIQR" if item=="DiffIQR" else item for item in methods_name]
df.index = methods_name

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, figsize=(8, 6))
ax.set_xlabel("Number of outliers (within 20 case samples)")
ax.set_ylabel("F1 scores of each method")
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])

ax = df_mixed.T.plot.bar(rot=0, legend=False, figsize=(8, 6))
ax.set_xlabel("Number of outliers (in case and control samples)")
ax.set_ylabel("F1 scores of each method")
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])

# Features
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated_normal_methods_features.csv"), 
                 sep=',', index_col=0)
data_name = df.columns.to_list()
methods_name = df.index.to_list()
methods_name = ["ΔIQR" if item=="DiffIQR" else item for item in methods_name]
df.index = methods_name

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, figsize=(8, 6))
ax.set_xlabel("Number of outliers (within 20 case samples)")
ax.set_ylabel("Number of significant features found by each method")
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])

ax = df_mixed.T.plot.bar(rot=0, legend=False, figsize=(8, 6))
ax.set_xlabel("Number of outliers (in case and control samples)")
ax.set_ylabel("Number of significant features found by each method")
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])

# Legend
ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10))
ax.set_xlabel("Number of outliers (in case and control samples)")
ax.set_ylabel("Number of significant features found by each method")
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.legend(title="Methods", title_fontsize=30, fontsize=25, ncol=5, 
          loc="lower right", bbox_to_anchor=(1.0, 1.0), 
          facecolor="None")