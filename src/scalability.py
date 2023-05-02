import os
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
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

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma",
           "KS", "KS+Gamma", "LIMMA", "LIMMA+Gamma", "HVF (composite)",
           "HVF (by condition)", "ΔHVF", "ΔHVF+ΔMean", "IQR (composite)",
           "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean", "COPA", "OS", "ORT",
           "MOST", "LSOSS", "DIDS", "DECO", "PHet (ΔHVF)", "PHet"]

# Define colors
PALETTE = sns.color_palette("tab20")
PALETTE.append("#fcfc81")
PALETTE.append("#C724B1")
PALETTE = dict([(item, mcolors.to_hex(PALETTE[idx])) for idx, item in enumerate(METHODS)])


def train():
    # Arguments
    direction = "both"
    num_iterations = 3
    # datasets: srbct, lung, baron, and pulseseq
    file_name = "srbct"
    suptitle_name = "srbct"
    expression_file_name = file_name + "_matrix.mtx"
    classes_file_name = file_name + "_classes.csv"
    alpha = 0.01

    # Load expression data and class data
    y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    num_examples, num_features = X.shape

    print("## Perform experimental studies using {0} data...".format(file_name))
    print("\t >> Sample size: {0}; Feature size: {1}".format(num_examples, num_features))
    current_progress = 1
    total_progress = len(METHODS) * num_iterations
    list_times = list()
    for iteration in range(num_iterations):
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[0]), end="\r")
        estimator = StudentTTest(use_statistics=False, direction=direction, adjust_pvalue=True, 
                                 adjusted_alpha=alpha)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[1]), end="\r")
        estimator = StudentTTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                 adjusted_alpha=alpha)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[2]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=False, direction=direction, adjust_pvalue=True, 
                                        adjusted_alpha=alpha)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[3]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                        adjusted_alpha=alpha)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[4]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                          adjusted_alpha=alpha)
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[5]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                          adjusted_alpha=alpha)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[6]), end="\r")
        estimator = SeuratHVF(per_condition=False, log_transform=True, num_top_features=num_features,
                              min_disp=0.5, min_mean=0.0125, max_mean=3)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[7]), end="\r")
        estimator = SeuratHVF(per_condition=True, log_transform=True, num_top_features=num_features,
                              min_disp=0.5, min_mean=0.0125, max_mean=3)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[8]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=True, num_top_features=num_features,
                                 min_disp=0.5, min_mean=0.0125, max_mean=3)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[9]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=True, num_top_features=num_features,
                                 min_disp=0.5, min_mean=0.0125, max_mean=3)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[10]), end="\r")
        estimator = HIQR(per_condition=False, normalize="zscore", iqr_range=(25, 75))
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[11]), end="\r")
        estimator = HIQR(per_condition=True, normalize="zscore", iqr_range=(25, 75))
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[12]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=False, normalize="zscore", iqr_range=(25, 75))
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[13]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=True, normalize="zscore", iqr_range=(25, 75))
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[14]), end="\r")
        estimator = COPA(q=75)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[15]), end="\r")
        estimator = OutlierSumStatistic(q=75, iqr_range=(25, 75), two_sided_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[16]), end="\r")
        estimator = OutlierRobustTstatistic(q=75, iqr_range=(25, 75))
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[17]), end="\r")
        estimator = MOST()
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[18]), end="\r")
        estimator = LSOSS(direction=direction)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[19]), end="\r")
        estimator = DIDS(score_function="tanh", direction=direction)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[20]), end="\r")
        estimator = PHeT(normalize=None, iqr_range=(25, 75), num_subsamples=1000, delta_type="hvf",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        if total_progress == current_progress:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                    METHODS[21]))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                    METHODS[21]), end="\r")
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

    list_times = np.reshape(list_times, (num_iterations, len(METHODS)))
    df = pd.DataFrame(list_times, index=range(num_iterations), columns=METHODS)
    df.index.name = 'Iterations'
    df = pd.melt(df.reset_index(), id_vars='Iterations', value_vars=METHODS, var_name="Methods",
                 value_name="Times")
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_times.csv"), sep=",", index=False)
    df = pd.read_csv(os.path.join(RESULT_PATH, file_name + "_times.csv"), sep=',')

    # Plot boxplot
    print("## Plot boxplot using top k features...")
    plt.figure(figsize=(14, 8))
    sns.boxplot(y='Times', x='Methods', data=df, width=0.85,
                palette=PALETTE)
    sns.swarmplot(y='Times', x='Methods', data=df, color="black", s=10, linewidth=0,
                  alpha=.7)
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
