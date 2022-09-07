import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import sort_features, comparative_score

sns.set_theme()


def train():
    # Arguments
    top_k_features = 100
    direction = "both"
    num_batches = 100
    subsample_size = 10

    # Load expression data
    X_control = pd.read_csv(os.path.join(DATASET_PATH, "her2_negative_matrix.csv"), sep=',')
    X_case = pd.read_csv(os.path.join(DATASET_PATH, "her2_positive_matrix.csv"), sep=',')
    features_name = X_control.columns.to_list()
    X_control = X_control.to_numpy()
    X_case = X_case.to_numpy()
    lb = LabelBinarizer()
    lb.fit(y=features_name)

    # Load top k features that are differentially expressed
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, "her2_topfeatures.csv"), sep=',')
    top_features_true = top_features_true["ID"].tolist()[:top_k_features]
    top_features_true = lb.transform(top_features_true).sum(axis=0).astype(int)

    print("## Perform simulation studies using HER2 data...")
    methods = ["COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "U-Het (zscore)", "U-Het (robust)"]
    list_scores = list()
    current_progress = 0
    total_progress = num_batches * len(methods)
    for batch_idx in range(num_batches):
        temp = np.random.choice(a=X_case.shape[0], size=subsample_size, replace=False)
        X = np.vstack((X_control, X_case[temp]))
        y = np.array(X_control.shape[0] * [0] + subsample_size * [1])

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "COPA"),
              end="\r")
        estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "OS"),
              end="\r")
        estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False,
                                        direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "ORT"),
              end="\r")
        estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "MOST"),
              end="\r")
        estimator = MOST(direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "LSOSS"),
              end="\r")
        estimator = LSOSS(direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "DIDS"),
              end="\r")
        estimator = DIDS(score_function="quad", direction=direction, calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                "U-Het (zscore)"), end="\r")
        estimator = UHeT(normalize="zcore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

        current_progress += 1
        if total_progress == current_progress:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    "U-Het (robust)"))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    "U-Het (robust)"), end="\r")
        estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75), calculate_pval=False)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:top_k_features].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(temp)

    list_scores = np.array(list_scores)
    list_scores = np.reshape(list_scores, (num_batches, len(methods)))

    # Plot boxplot
    print("## Plot boxplot using top k features...")
    df = pd.DataFrame(list_scores, index=range(num_batches), columns=methods)
    df.index.name = 'Batch'
    df = pd.melt(df.reset_index(), id_vars='Batch', value_vars=methods, var_name="Methods",
                 value_name="Jaccard scores")
    plt.figure(figsize=(12, 8))
    bplot = sns.boxplot(y='Jaccard scores', x='Methods', data=df, width=0.5,
                        palette="tab10")
    bplot = sns.swarmplot(y='Jaccard scores', x='Methods', data=df,
                          color='black', alpha=0.75)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Method', fontsize=20, fontweight="bold")
    plt.ylabel("Jaccard scores", fontsize=20, fontweight="bold")
    plt.suptitle("Results using Her2 data for {0} batches".format(num_batches),
                 fontsize=22, fontweight="bold")
    sns.despine()
    file_path = os.path.join(RESULT_PATH, "her2.png")
    plt.tight_layout()
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
    train()
