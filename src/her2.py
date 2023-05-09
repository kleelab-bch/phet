import os
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelBinarizer

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
from utility.utils import sort_features

sns.set_theme()
sns.set_style(style='white')

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma",
           "KS", "KS+Gamma", "LIMMA", "LIMMA+Gamma", "Dispersion (composite)",
           "Dispersion (by condition)", "ΔDispersion", "ΔDispersion+ΔMean",
           "IQR (composite)", "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean",
           "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "PHet (ΔDispersion)",
           "PHet"]

# Define colors
PALETTE = sns.color_palette("tab20")
PALETTE.append("#fcfc81")
PALETTE.append("#C724B1")
PALETTE.append("#fcfc81")
PALETTE.append("#b5563c")
PALETTE.append("#C724B1")
PALETTE.append("#606c38")
PALETTE.append("#283618")
PALETTE = dict([(item, mcolors.to_hex(PALETTE[idx])) for idx, item in enumerate(METHODS)])


def compute_score(lb, top_features_true, top_features_pred, probes2genes, genes2probes,
                  range_topfeatures):
    top_features_pred = top_features_pred["features"].to_list()
    temp_range = list()
    for top_features in range_topfeatures:
        temp = top_features_pred[:top_features]
        add_probes = [p for probe in probes2genes if probe in temp
                      for p in genes2probes[probes2genes[probe]]]
        temp.extend(add_probes)
        temp = list(set(temp))
        if len(temp) != 0:
            pred_features = lb.transform(temp).sum(axis=0).astype(int)
            temp = comparative_score(pred_features=pred_features, true_features=top_features_true,
                                     metric="precision")
        else:
            temp = 0
        temp_range.append(temp)
    return temp_range


def single_batch(X_case, X_control, X_deco, X_limma, X_limma_distr, top_features_true, features_name, genes2probes,
                 probes2genes, lb, range_topfeatures, subsample_size, alpha, direction, batch_idx, total_progress):
    desc = "\t\t--> Progress: {0:.4f}%".format(((batch_idx + 1) / total_progress) * 100)
    if total_progress == batch_idx + 1:
        print(desc)
    else:
        print(desc, end="\r")

    list_scores = list()
    temp = np.random.choice(a=X_case.shape[0], size=subsample_size, replace=False)
    X = np.vstack((X_control, X_case[temp]))
    y = np.array(X_control.shape[0] * [0] + subsample_size * [1])
    num_features = X.shape[1]

    estimator = StudentTTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                             adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False, ascending=True)
    top_features_pred = top_features_pred[top_features_pred["score"] < alpha]
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = StudentTTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                             adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = WilcoxonRankSumTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                    adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False, ascending=True)
    top_features_pred = top_features_pred[top_features_pred["score"] < alpha]
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = WilcoxonRankSumTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                    adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = KolmogorovSmirnovTest(use_statistics=False, direction=direction, adjust_pvalue=True,
                                      adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False, ascending=True)
    top_features_pred = top_features_pred[top_features_pred["score"] < alpha]
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = KolmogorovSmirnovTest(use_statistics=True, direction=direction, adjust_pvalue=True,
                                      adjusted_alpha=alpha)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    top_features_pred = X_limma
    temp = np.nonzero(top_features_pred)[0]
    top_features_pred[temp] = np.max(top_features_pred) + 1 - top_features_pred[temp]
    top_features_pred = top_features_pred.reshape(top_features_pred.shape[0], 1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    top_features_pred = X_limma_distr
    top_features_pred = top_features_pred.reshape(top_features_pred.shape[0], 1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = SeuratHVF(per_condition=False, log_transform=False, num_top_features=num_features,
                          min_disp=0.5, min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    top_features_pred = estimator.fit_predict(X=temp_X, y=y, control_class=0, case_class=1)
    del temp_X
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = SeuratHVF(per_condition=True, log_transform=False, num_top_features=num_features,
                          min_disp=0.5, min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    top_features_pred = estimator.fit_predict(X=temp_X, y=y, control_class=0, case_class=1)
    del temp_X
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=False, num_top_features=num_features,
                             min_disp=0.5, min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    top_features_pred = estimator.fit_predict(X=temp_X, y=y, control_class=0, case_class=1)
    del temp_X
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=False, num_top_features=num_features,
                             min_disp=0.5, min_mean=0.0125, max_mean=3)
    temp_X = deepcopy(X)
    top_features_pred = estimator.fit_predict(X=temp_X, y=y, control_class=0, case_class=1)
    del temp_X
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = HIQR(per_condition=False, normalize="zscore", iqr_range=(25, 75))
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = HIQR(per_condition=True, normalize="zscore", iqr_range=(25, 75))
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = DeltaIQRMean(calculate_deltamean=False, normalize="zscore", iqr_range=(25, 75))
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = DeltaIQRMean(calculate_deltamean=True, normalize="zscore", iqr_range=(25, 75))
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = COPA(q=75)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = OutlierSumStatistic(q=75, iqr_range=(25, 75), two_sided_test=False)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = OutlierRobustTstatistic(q=75, iqr_range=(25, 75))
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = MOST()
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = LSOSS(direction=direction)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = DIDS(score_function="tanh", direction=direction)
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    top_features_pred = X_deco
    temp = np.nonzero(top_features_pred)[0]
    top_features_pred[temp] = np.max(top_features_pred) + 1 - top_features_pred[temp]
    top_features_pred = top_features_pred.reshape(top_features_pred.shape[0], 1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = PHeT(normalize=None, iqr_range=(25, 75), num_subsamples=1000, delta_type="hvf",
                     calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                     calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                     weight_range=[0.2, 0.4, 0.8])
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                     calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                     calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                     weight_range=[0.2, 0.4, 0.8])
    top_features_pred = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    top_features_pred = sort_features(X=top_features_pred, features_name=features_name,
                                      X_map=None, map_genes=False)
    temp_range = compute_score(lb=lb, top_features_true=top_features_true,
                               top_features_pred=top_features_pred,
                               probes2genes=probes2genes, genes2probes=genes2probes,
                               range_topfeatures=range_topfeatures)
    list_scores.append(temp_range)

    return list_scores


def train(num_jobs: int = 2):
    # Filtering and subsampling arguments
    alpha = 0.01
    top_k_features = 500
    range_topfeatures = list(range(0, top_k_features + 5, 5))
    range_topfeatures[0] = 1
    num_batches = 1000
    subsample_size = 10

    # Models parameters
    direction = "both"

    # Load expression data
    X_control = pd.read_csv(os.path.join(DATASET_PATH, "her2_negative_matrix.csv"), sep=',')
    X_case = pd.read_csv(os.path.join(DATASET_PATH, "her2_positive_matrix.csv"), sep=',')
    features_name = X_control.columns.to_list()
    X_control = X_control.to_numpy()
    X_case = X_case.to_numpy()
    lb = LabelBinarizer()
    lb.fit(y=features_name)

    # Load genes that are known to be encoded on chromosome 17
    X_humchr17 = pd.read_csv(os.path.join(DATASET_PATH, "humchr17.csv"), sep=',')
    temp = [idx for idx, item in enumerate(X_humchr17["Chromosomal position"].tolist())
            if item == "17q12" or item == "17q21.1"]
    X_humchr17 = X_humchr17.iloc[temp]["Gene name"].tolist()

    # Load top k features that are differentially expressed
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, "her2_topfeatures.csv"),
                                    sep=',', header=0)
    temp = [idx for idx, item in enumerate(top_features_true["Gene.symbol"])
            if item in X_humchr17 and top_features_true.iloc[idx]["adj.P.Val"] <= 0.01]
    selected_features = np.unique(top_features_true.iloc[temp]["Gene.symbol"].tolist()).shape[0]
    probes2genes = {}
    genes2probes = {}
    for item in top_features_true.iloc[temp]["ID"].to_list():
        probes2genes[item] = top_features_true[top_features_true["ID"] == item]["Gene.symbol"].to_list()[0]
    for probe, gene in probes2genes.items():
        if gene in genes2probes:
            genes2probes[gene] += [probe]
        else:
            genes2probes[gene] = [probe]
    genes2probes = dict((gene, probe) for gene, probe in genes2probes.items()
                        if len(probe) > 1)
    probes2genes = dict([(probe, probes2genes[probe]) for probes in genes2probes.values()
                         for probe in probes])
    top_features_true = top_features_true.iloc[temp]["ID"].tolist()
    top_features_true = lb.transform(top_features_true).sum(axis=0).astype(int)
    top_k_features = sum(top_features_true).astype(int)

    # Load DECO and LIMMA results    
    X_deco = pd.read_csv(os.path.join(DATASET_PATH, "her2_deco_features.csv"), sep=',', header=None)
    X_deco = X_deco.to_numpy()
    if X_deco.shape[1] != num_batches:
        temp = "The number of batches does not match with DECO results"
        raise Exception(temp)
    X_limma = pd.read_csv(os.path.join(DATASET_PATH, "her2_limma_features.csv"), sep=',', header=None)
    X_limma = X_limma.to_numpy()
    if X_limma.shape[1] != num_batches:
        temp = "The number of batches does not match with LIMMA results"
        raise Exception(temp)
    X_limma_distr = pd.read_csv(os.path.join(DATASET_PATH, "her2_limma_distr_features.csv"), sep=',', header=None)
    X_limma_distr = np.absolute(X_limma_distr.to_numpy())
    if X_limma_distr.shape[1] != num_batches:
        temp = "The number of batches does not match with LIMMA results"
        raise Exception(temp)

    print("## Perform simulation studies using HER2 data...")
    print(
        "\t >> Control size: {0}; Case size: {1}; Feature size: {2}; True feature size: {3}".format(X_control.shape[0],
                                                                                                    X_case.shape[0],
                                                                                                    len(features_name),
                                                                                                    selected_features))
    parallel = Parallel(n_jobs=num_jobs, prefer="threads", verbose=0)
    list_scores = parallel(delayed(single_batch)(X_case, X_control, X_deco[:, batch_idx], X_limma[:, batch_idx],
                                                 X_limma_distr[:, batch_idx], top_features_true, features_name,
                                                 genes2probes, probes2genes, lb, range_topfeatures, subsample_size,
                                                 alpha, direction, batch_idx, num_batches)
                           for batch_idx in range(num_batches))
    list_scores = [i for item in list_scores for i in item]
    list_scores = np.array(list_scores)
    list_scores = np.reshape(list_scores, (num_batches, len(METHODS) * len(range_topfeatures)))
    # Transform to dataframe
    temp_methods = [m for m in METHODS for f in range_topfeatures]
    df = pd.DataFrame(list_scores, index=range(num_batches), columns=temp_methods)
    df.index.name = 'Batch'
    df = pd.melt(df.reset_index(), id_vars='Batch', value_vars=temp_methods, var_name="Methods",
                 value_name="Scores")
    temp_range = [f for m in METHODS for f in range_topfeatures for b in range(num_batches)]
    temp_range = pd.Series(temp_range)
    df["Range"] = temp_range
    df.to_csv(os.path.join(RESULT_PATH, "her2_scores.csv"), sep=',', index=False)

    # TODO:delete below
    # df = pd.read_csv(os.path.join(RESULT_PATH, "her2_scores.csv"), sep=',')
    # temp = [idx for idx, item in enumerate(df["Methods"].tolist())]
    # df = df.iloc[temp]

    # Plot lineplot
    print("## Plot lineplot using top k features...")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='Range', y='Scores', hue="Methods", palette=PALETTE, style="Methods")
    plt.axvline(20, color='red')
    plt.xticks([item for item in range_topfeatures if item % 25 == 0 or item == 1], fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.xlabel('Top k features', fontsize=22)
    plt.ylabel("F1 scores of each method", fontsize=22)
    plt.suptitle("Results using Her2 data", fontsize=26)
    sns.despine()
    plt.tight_layout()
    file_path = os.path.join(RESULT_PATH, "her2_lineplot.png")
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
    train(num_jobs=25)
