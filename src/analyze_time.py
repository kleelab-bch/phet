import os
import time

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


def train(num_jobs: int = 4):
    # Arguments
    direction = "both"
    minimum_samples = 5
    num_iterations = 1000
    methods = ["t-statistic", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "ΔIQR", "PHet"]

    # 1. Micro-array datasets: allgse412, amlgse2191, bc_ccgse3726, bcca1, bcgse349_350, bladdergse89,
    # braintumor, cmlgse2535, colon, dlbcl, ewsgse967, gastricgse2685, glioblastoma, leukemia_golub, 
    # ll_gse1577_2razreda, lung, lunggse1987, meduloblastomigse468, mll, myelodysplastic_mds1, 
    # myelodysplastic_mds2, pdac, prostate, prostategse2443, srbct, and tnbc
    # 2. scRNA datasets: camp2, darmanis, lake, yan, camp1, baron, segerstolpe, wang, li, and patel
    file_name = "patel"
    expression_file_name = file_name + "_matrix"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, expression_file_name + ".csv"), sep=',').dropna(axis=1)
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter data based on counts (CPM)
    example_sums = np.absolute(X).sum(1)
    examples_ids = np.where(example_sums >= 5)[0]
    X = X[examples_ids]
    y = y[examples_ids]
    num_examples, num_features = X.shape
    del example_sums, examples_ids
    temp = np.absolute(X)
    temp = (temp * 1e6) / temp.sum(axis=1).reshape((num_examples, 1))
    temp[temp > 1] = 1
    temp[temp != 1] = 0
    feature_sums = temp.sum(0)
    if num_examples <= minimum_samples or minimum_samples > num_examples // 2:
        minimum_samples = num_examples // 2
    feature_ids = np.where(feature_sums >= minimum_samples)[0]
    features_name = np.array(features_name)[feature_ids].tolist()
    X = X[:, feature_ids]
    feature_ids = dict([(feature_idx, idx) for idx, feature_idx in enumerate(feature_ids)])
    num_examples, num_features = X.shape
    del temp, feature_sums

    print("## Perform experimental studies using {0} data...".format(file_name))
    print("\t >> Sample size: {0}; Feature size: {1}".format(X.shape[0], X.shape[1]))
    current_progress = 1
    total_progress = len(methods) * num_iterations
    list_times = list()
    for iteration in range(num_iterations):
        if current_progress == total_progress:
            print("\t >> Progress: {0:.4f}%".format((current_progress / total_progress) * 100))
        else:
            print("\t >> Progress: {0:.4f}%".format((current_progress / total_progress) * 100), end="\r")

        estimator = StudentTTest(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = COPA(q=0.75, direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                        permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                           permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = MOST(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = LSOSS(direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = DIDS(score_function="tanh", direction=direction, permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = DeltaIQR(normalize="zscore", q=0.75, iqr_range=(25, 75), permutation_test=False)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y)
        list_times.append(time.time() - curr_time)
        current_progress += 1

        estimator = PHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=1000, subsampling_size=None,
                         alpha_subsample=0.05, partition_by_anova=False, bin_KS_pvalues=False,
                         feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.1, 0.4, 0.8], calculate_hstatistic=False, num_components=10,
                         num_subclusters=10,
                         binary_clustering=True, permutation_test=False, num_rounds=50, num_jobs=num_jobs)
        curr_time = time.time()
        estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        list_times.append(time.time() - curr_time)
        current_progress += 1
    list_times = np.array(list_times).reshape((len(methods), num_iterations))
    df = pd.DataFrame(list_times, index=methods)
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, file_name + "_times.csv"), sep=",")


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
