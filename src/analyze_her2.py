import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

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
from utility.utils import sort_features, comparative_score

sns.set_theme()
sns.set_style(style='white')


def train(num_jobs: int = 4):
    # Arguments
    direction = "both"
    topKfeatures = 100
    calculate_hstatistic = False
    num_batches = 1000
    subsample_size = 10
    methods_name = ["ttest", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "DeltaIQR", "PHet"]

    # Load expression data
    X_control = pd.read_csv(os.path.join(DATASET_PATH, "her2_negative_matrix.csv"), sep=',')
    X_case = pd.read_csv(os.path.join(DATASET_PATH, "her2_positive_matrix.csv"), sep=',')
    features_name = X_control.columns.to_list()
    X_control = X_control.to_numpy()
    X_case = X_case.to_numpy()
    lb = LabelBinarizer()
    lb.fit(y=features_name)

    # Load genes genes that are known to be encoded on chromosome 17
    X_humchr17 = pd.read_csv(os.path.join(DATASET_PATH, "humchr17.csv"), sep=',')
    temp = [idx for idx, item in enumerate(X_humchr17["Chromosomal position"].tolist())
            if item == "17q12" or item == "17q21.1"]
    X_humchr17 = X_humchr17.iloc[temp]["Gene name"].tolist()

    # Load top k features that are differentially expressed
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, "her2_topfeatures.csv"), sep=',')
    temp = [idx for idx, item in enumerate(top_features_true["Gene.symbol"])
            if item in X_humchr17 and idx <= topKfeatures]
    top_features_true = top_features_true.iloc[temp]["ID"].tolist()
    top_features_true = lb.transform(top_features_true).sum(axis=0).astype(int)
    topKfeatures = sum(top_features_true).astype(int)

    # Load DECO results    
    df_deco = pd.read_csv(os.path.join(DATASET_PATH, "her2_deco.csv"), sep=',', header=None)
    df_deco = df_deco.to_numpy()
    if df_deco.shape[1] != num_batches:
        temp = "The number of bacthes does not macth with DECO results"
        raise Exception(temp)

    print("## Perform simulation studies using HER2 data...")
    print("\t >> Control size: {0}; Case size: {1}; Feature size: {2}".format(X_control.shape[0], X_case.shape[1],
                                                                              len(features_name)))
    list_scores = list()
    current_progress = 1
    total_progress = num_batches * len(methods_name)
    for batch_idx in range(num_batches):
        temp = np.random.choice(a=X_case.shape[0], size=subsample_size, replace=False)
        X = np.vstack((X_control, X_case[temp]))
        y = np.array(X_control.shape[0] * [0] + subsample_size * [1])

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                "Studentsttest"),
              end="\r")
        estimator = StudentTTest(direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "COPA"),
              end="\r")
        estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "OS"),
              end="\r")
        estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False,
                                        direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "ORT"),
              end="\r")
        estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "MOST"),
              end="\r")
        estimator = MOST(direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "LSOSS"),
              end="\r")
        estimator = LSOSS(direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "DIDS"),
              end="\r")
        estimator = DIDS(score_function="tanh", direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "DECO"),
              end="\r")
        top_features_pred = df_deco[:, batch_idx]
        temp = np.nonzero(top_features_pred)[0]
        top_features_pred[temp] = np.max(top_features_pred) + 1 - top_features_pred[temp]
        top_features_pred = top_features_pred.reshape(top_features_pred.shape[0], 1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                "DeltaIQR"), end="\r")
        estimator = DeltaIQR(normalize="zcore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        if total_progress == current_progress:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    "PHet"))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    "PHet"), end="\r")
        estimator = PHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=1000, subsampling_size=None,
                         significant_p=0.05, partition_by_anova=False, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.1, 0.3, 0.5], calculate_hstatistic=calculate_hstatistic,
                         num_components=10, num_subclusters=10, binary_clustering=True, calculate_pval=False,
                         num_rounds=50, num_jobs=num_jobs)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

    list_scores = np.array(list_scores)
    list_scores = np.reshape(list_scores, (num_batches, len(methods_name)))

    # Transform to dataframe
    df = pd.DataFrame(list_scores, index=range(num_batches), columns=methods_name)
    df.index.name = 'Batch'
    df = pd.melt(df.reset_index(), id_vars='Batch', value_vars=methods_name, var_name="Methods",
                 value_name="F1 scores")
    df.to_csv(os.path.join(RESULT_PATH, "her2_scores.csv"), sep=',', index=False)
    df = pd.read_csv(os.path.join(RESULT_PATH, "her2_scores.csv"), sep=',')
    temp = [idx for idx, item in enumerate(df["Methods"].tolist()) if item != "DECO"]
    df = df.iloc[temp]
    temp = ["ΔIQR" if item == "DeltaIQR" else item for item in df["Methods"].tolist()]
    df["Methods"] = temp
    palette = mcolors.TABLEAU_COLORS
    methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
    palette = dict([(methods_name[idx], item[1]) for idx, item in enumerate(palette.items())
                    if idx + 1 <= len(methods_name) and methods_name[idx] != "DECO"])

    # Plot boxplot
    print("## Plot boxplot using top k features...")
    plt.figure(figsize=(14, 8))
    bplot = sns.boxplot(y='F1 scores', x='Methods', data=df, width=0.5,
                        palette=palette)
    # bplot = sns.swarmplot(y='F1 scores', x='Methods', data=df,
    #                      color='black', alpha=0.75)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Methods', fontsize=22)
    plt.ylabel("F1 scores of each method", fontsize=22)
    plt.suptitle("Results using Her2 data", fontsize=26)
    sns.despine()
    plt.tight_layout()
    file_path = os.path.join(RESULT_PATH, "her2_boxplot.png")
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)

result_path = os.path.join(RESULT_PATH, "scRNA")
files = [file for file in os.listdir(result_path) if file.endswith("_features_scores.csv")]
# files = [file for file in os.listdir(result_path) if file.endswith("_cluster_accuracy.csv")]
methods_name = []
methods = []
scores = []
for f in files:
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    methods_name = df.iloc[:, 0].to_list()
    methods.extend(df.iloc[:, 0].to_list())
    scores.extend(df.iloc[:, 1].to_list())

methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
methods = ["ΔIQR" if item == "DeltaIQR" else item for item in methods]
df = pd.DataFrame([methods, scores]).T
df.columns = ["Methods", "F1 scores"]
mean_scores = np.mean(scores, 0)
std_scores = np.std(scores, 0)

palette = mcolors.TABLEAU_COLORS
palette = dict([(methods_name[idx], item[1]) for idx, item in enumerate(palette.items())
                if idx + 1 <= len(methods_name)])

plt.figure(figsize=(14, 8))
bplot = sns.boxplot(y='F1 scores', x='Methods', data=df, palette=palette)
plt.xticks(fontsize=20, rotation=45)
plt.yticks(fontsize=20)
plt.xlabel('Methods', fontsize=24)
plt.ylabel("F1 scores of each method", fontsize=26)
plt.suptitle("Results using 6 scRNA datasets", fontsize=28)
sns.despine()
plt.tight_layout()