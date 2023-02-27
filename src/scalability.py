import os
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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

sns.set_theme()
sns.set_theme(style="white")


def train():
    # Arguments
    direction = "both"
    num_iterations = 10
    methods = ["t-statistic", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "ΔIQR", "PHet"]

    # datasets: srbct, lung, baron, and pulseseq
    file_name = "lung"
    suptitle_name = "Lung"
    expression_file_name = file_name + "_matrix"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, expression_file_name + ".csv"), sep=',').dropna(axis=1)
    y = X["class"].to_numpy()
    X = X.drop(["class"], axis=1).to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    num_examples, num_features = X.shape

    print("## Perform experimental studies using {0} data...".format(file_name))
    print("\t >> Sample size: {0}; Feature size: {1}".format(num_examples, num_features))
    current_progress = 1
    total_progress = len(methods) * num_iterations
    list_times = list()
    for iteration in range(num_iterations):
        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[0]), end="\r")
        estimator = StudentTTest(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[1]), end="\r")
        estimator = COPA(q=0.75, direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[2]), end="\r")
        estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                        permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[3]), end="\r")
        estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                           permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[4]), end="\r")
        estimator = MOST(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[5]), end="\r")
        estimator = LSOSS(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[6]), end="\r")
        estimator = DIDS(score_function="tanh", direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                methods[7]), end="\r")
        estimator = DeltaIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        if total_progress == current_progress:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    methods[7]))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                                    methods[7]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, alpha_subsample=0.05,
                         calculate_deltaiqr=True, calculate_fisher=True, calculate_profile=True,
                         calculate_hstatistic=False, bin_KS_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.1, 0.4, 0.8])
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

    list_times = np.reshape(list_times, (num_iterations, len(methods)))
    df = pd.DataFrame(list_times, index=range(num_iterations), columns=methods)
    df.index.name = 'Iterations'
    df = pd.melt(df.reset_index(), id_vars='Iterations', value_vars=methods, var_name="Methods",
                 value_name="Times")
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_times.csv"), sep=",", index=False)
    df = pd.read_csv(os.path.join(RESULT_PATH, file_name + "_times.csv"), sep=',')
    palette = mcolors.TABLEAU_COLORS
    palette = dict([(methods[idx], item[1]) for idx, item in enumerate(palette.items())
                    if idx + 1 <= len(methods)])

    # Plot boxplot
    print("## Plot boxplot using top k features...")
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Methods', y='Times', width=0.5, palette=palette)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.xlabel('Methods', fontsize=22)
    plt.ylabel("Times", fontsize=22)
    plt.suptitle("Results using {0} data".format(suptitle_name), fontsize=26)
    sns.despine()
    plt.tight_layout()
    file_path = os.path.join(RESULT_PATH, file_name + "_times_boxplot.png")
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
