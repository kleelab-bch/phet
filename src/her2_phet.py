import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

from model.phet import PHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.utils import sort_features, comparative_score

sns.set_theme()
sns.set_style(style='white')


def train():
    # Arguments
    feature_weight = [0.4, 0.3, 0.2, 0.1]
    weight_range = [0.1, 0.4, 0.8]
    topKfeatures = 100
    num_batches = 1000
    subsample_size = 10
    bin_KS_pvalues = True
    if bin_KS_pvalues:
        phet_name = "phet_b"
    else:
        phet_name = "phet_nb"
    methods = ["PHet (ΔIQR)", "PHet (Fisher)", "PHet (Profile)", "PHet (ΔIQR+Fisher)",
               "PHet (ΔIQR+Profile)", "PHet (Fisher+Profile)", "PHet (no Binning)",
               "PHet"]

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

    print("## Perform simulation studies using HER2 data...")
    print("\t >> Control size: {0}; Case size: {1}; Feature size: {2}".format(X_control.shape[0], X_case.shape[0],
                                                                              len(features_name)))
    list_scores = list()
    current_progress = 1
    total_progress = num_batches * len(methods)
    for batch_idx in range(num_batches):
        temp = np.random.choice(a=X_case.shape[0], size=subsample_size, replace=False)
        X = np.vstack((X_control, X_case[temp]))
        y = np.array(X_control.shape[0] * [0] + subsample_size * [1])

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[0]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=False, calculate_profile=False,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[1]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=False, calculate_fisher=True, calculate_profile=False,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[2]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=False, calculate_fisher=False, calculate_profile=True,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[3]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=False,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[4]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=False, calculate_profile=True,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[5]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=False, calculate_fisher=True, calculate_profile=True,
                         bin_KS_pvalues=bin_KS_pvalues, feature_weight=feature_weight,
                         weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                methods[6]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=True,
                         bin_KS_pvalues=False, feature_weight=feature_weight, weight_range=weight_range)
        top_features_pred = estimator.fit_predict(X=X, y=y)
        top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                          X_map=None, map_genes=False)
        top_features_pred = top_features_pred["features"][:topKfeatures].to_list()
        top_features_pred = lb.transform(top_features_pred).sum(axis=0).astype(int)
        temp = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                 metric="f1")
        list_scores.append(temp)
        current_progress += 1

        if current_progress == total_progress:
            print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                    methods[7]))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:35}".format((current_progress / total_progress) * 100,
                                                                    methods[7]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=True,
                         bin_KS_pvalues=True, feature_weight=feature_weight, weight_range=weight_range)
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
    list_scores = np.reshape(list_scores, (num_batches, len(methods)))

    # Transform to dataframe
    df = pd.DataFrame(list_scores, index=range(num_batches), columns=methods)
    df.index.name = 'Batch'
    df = pd.melt(df.reset_index(), id_vars='Batch', value_vars=methods, var_name="Methods",
                 value_name="Scores")
    df.to_csv(os.path.join(RESULT_PATH, "her2_" + phet_name + "_scores.csv"), sep=',', index=False)
    df = pd.read_csv(os.path.join(RESULT_PATH, "her2_" + phet_name + "_scores.csv"), sep=',')
    palette = mcolors.TABLEAU_COLORS
    palette = dict([(methods[idx], item[1]) for idx, item in enumerate(palette.items())
                    if idx + 1 <= len(methods)])

    # Plot boxplot
    print("## Plot boxplot using top k features...")
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Methods', y='Scores', width=0.5, palette=palette)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Methods', fontsize=22)
    plt.ylabel("F1 scores of each method", fontsize=22)
    plt.suptitle("Results using Her2 data", fontsize=26)
    sns.despine()
    plt.tight_layout()
    file_path = os.path.join(RESULT_PATH, "her2_" + phet_name + "_b_boxplot.png")
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
