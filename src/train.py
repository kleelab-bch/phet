import os
from copy import deepcopy

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
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma", "KS", "KS+Gamma", "LIMMA",
           "LIMMA+Gamma", "Dispersion (composite)", "Dispersion (by condition)", "ΔDispersion",
           "ΔDispersion+ΔMean", "IQR (composite)", "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean", "COPA",
           "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO", "PHet (ΔDispersion)", "PHet"]


def train(args):
    # Set up the number of operations to apply
    steps = 1

    # Filtering and global arguments
    minimum_samples = 5
    temp_methods = []
    methods_save_name = []

    ##########################################################################################################
    ###################                  Load and preprocess omics data                   ####################
    ##########################################################################################################
    print('\n{0})- Load and preprocess {1} dataset...'.format(steps, args.file_name))
    steps = steps + 1

    # Expression, classes, subtypes, donors, timepoints files
    expression_file_name = args.file_name + "_matrix.mtx"
    features_file_name = args.file_name + "_feature_names.csv"
    classes_file_name = args.file_name + "_classes.csv"
    subtypes_file = args.file_name + "_types.csv"
    differential_features_file = args.file_name + "_limma_features.csv"

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(args.dspath, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))

    # Load features, expression, and class data
    features_name = pd.read_csv(os.path.join(args.dspath, features_file_name), sep=',')
    features_name = features_name["features"].to_list()
    y = pd.read_csv(os.path.join(args.dspath, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(args.dspath, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter data
    num_examples, num_features = X.shape
    example_sums = np.absolute(X).sum(1)
    examples_ids = np.where(example_sums >= 5)[0]  # filter out cells below 5
    X = X[examples_ids]
    y = y[examples_ids]
    subtypes = np.array(subtypes)[examples_ids].tolist()
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

    # Save subtypes for SPRING
    if args.export_spring:
        df = pd.DataFrame(subtypes, columns=["subtypes"]).T
        df.to_csv(os.path.join(args.rspath, args.file_name + "_subtypes.csv"), sep=',', header=False)
        del df

    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(args.dspath, differential_features_file), sep=',',
                                    index_col="ID")
    temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
    if top_features_true.shape[1] > 0:
        top_features_true = top_features_true.loc[temp]
        temp = top_features_true[top_features_true["adj.P.Val"] < args.alpha]
        if temp.shape[0] < args.top_k_features:
            temp = top_features_true[:args.top_k_features - 1]
            if args.sort_by_pvalue and temp.shape[0] == 0:
                plot_top_k_features = True
        top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:args.top_k_features]]
    else:
        top_features_true = temp
        args.top_k_features = len(top_features_true)
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    ##########################################################################################################
    #####################                            Inference                            ####################
    ##########################################################################################################

    print('\n{0})- Infer subtypes using {1} dataset...'.format(steps, args.file_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    steps = steps + 1
    current_progress = 1
    total_progress = len(args.methods)
    methods_dict = dict()

    if "ttest_p" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[0]), end="\r")
        estimator = StudentTTest(use_statistics=False, direction=args.direction, adjust_pvalue=True,
                                 adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] < args.alpha]
        methods_save_name.append("ttest_p")
        temp_methods.append(METHODS[0])
        methods_dict.update({METHODS[0]: df})
        current_progress += 1

    if "ttest_g" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[1]), end="\r")
        estimator = StudentTTest(use_statistics=True, direction=args.direction, adjust_pvalue=True,
                                 adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("ttest_g")
        temp_methods.append(METHODS[1])
        methods_dict.update({METHODS[1]: df})
        current_progress += 1

    if "wilcoxon_p" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[2]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=False, direction=args.direction, adjust_pvalue=True,
                                        adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] < args.alpha]
        methods_save_name.append("wilcoxon_p")
        temp_methods.append(METHODS[2])
        methods_dict.update({METHODS[2]: df})
        current_progress += 1

    if "wilcoxon_g" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[3]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=True, direction=args.direction, adjust_pvalue=True,
                                        adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("wilcoxon_g")
        temp_methods.append(METHODS[3])
        methods_dict.update({METHODS[3]: df})
        current_progress += 1

    if "ks_p" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[4]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=False, direction=args.direction, adjust_pvalue=True,
                                          adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] < args.alpha]
        methods_save_name.append("ks_p")
        temp_methods.append(METHODS[4])
        methods_dict.update({METHODS[4]: df})
        current_progress += 1

    if "ks_g" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[5]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=True, direction=args.direction, adjust_pvalue=True,
                                          adjusted_alpha=args.alpha)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("ks_g")
        temp_methods.append(METHODS[5])
        methods_dict.update({METHODS[5]: df})
        current_progress += 1

    if "limma_p" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[6]), end="\r")
        if not os.path.exists(os.path.join(args.dspath, args.file_name + "_limma_features.csv")):
            raise Exception("Please provide limma features file!")
        df = pd.read_csv(os.path.join(args.dspath, args.file_name + "_limma_features.csv"), sep=',')
        df = df[["ID", "adj.P.Val", "B"]]
        df = df[df["adj.P.Val"] < args.alpha]
        df = df[["ID", "B"]]
        df.columns = ["features", "score"]
        methods_save_name.append("limma_p")
        temp_methods.append(METHODS[6])
        methods_dict.update({METHODS[6]: df})
        current_progress += 1

    if "limma_g" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[7]), end="\r")
        if not os.path.exists(os.path.join(args.dspath, args.file_name + "_limma_features.csv")):
            raise Exception("Please provide limma features file!")
        df = pd.read_csv(os.path.join(args.dspath, args.file_name + "_limma_features.csv"), sep=',')
        df = df[["ID", "B"]]
        temp = [features_name.index(item) for item in df["ID"].to_list() if item in features_name]
        df = np.absolute(df.iloc[temp]["B"].to_numpy()[:, None])
        methods_save_name.append("limma_g")
        temp_methods.append(METHODS[7])
        methods_dict.update({METHODS[7]: df})
        current_progress += 1

    if "dispersion_a" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[8]), end="\r")
        estimator = SeuratHVF(per_condition=False, log_transform=args.seurat_log_transform,
                              num_top_features=num_features,
                              min_disp=0.5, min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        if args.exponentiate:
            temp_X = np.exp(temp_X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_save_name.append("dispersion_a")
        temp_methods.append(METHODS[8])
        methods_dict.update({METHODS[8]: df})
        current_progress += 1

    if "dispersion_c" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[9]), end="\r")
        estimator = SeuratHVF(per_condition=True, log_transform=args.seurat_log_transform,
                              num_top_features=num_features,
                              min_disp=0.5, min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        if args.exponentiate:
            temp_X = np.exp(temp_X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_save_name.append("dispersion_c")
        temp_methods.append(METHODS[9])
        methods_dict.update({METHODS[9]: df})
        current_progress += 1

    if "deltadispersion" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[10]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=args.seurat_log_transform,
                                 num_top_features=num_features, min_disp=0.5, min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        if args.exponentiate:
            temp_X = np.exp(temp_X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_save_name.append("deltadispersion")
        temp_methods.append(METHODS[10])
        methods_dict.update({METHODS[10]: df})
        current_progress += 1

    if "deltadispersionmean" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[11]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=args.seurat_log_transform,
                                 num_top_features=num_features, min_disp=0.5, min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        if args.exponentiate:
            temp_X = np.exp(temp_X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_save_name.append("deltadispersionmean")
        temp_methods.append(METHODS[11])
        methods_dict.update({METHODS[11]: df})
        current_progress += 1

    if "iqr_a" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[12]), end="\r")
        estimator = HIQR(per_condition=False, normalize=args.normalize, iqr_range=args.iqr_range)
        df = estimator.fit_predict(X=X, y=y)
        methods_save_name.append("iqr_a")
        temp_methods.append(METHODS[12])
        methods_dict.update({METHODS[12]: df})
        current_progress += 1

    if "iqr_c" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[13]), end="\r")
        estimator = HIQR(per_condition=True, normalize=args.normalize, iqr_range=args.iqr_range)
        df = estimator.fit_predict(X=X, y=y)
        methods_save_name.append("iqr_c")
        temp_methods.append(METHODS[13])
        methods_dict.update({METHODS[13]: df})
        current_progress += 1

    if "deltaiqr" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[14]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=False, normalize=args.normalize, iqr_range=args.iqr_range)
        df = estimator.fit_predict(X=X, y=y)
        methods_save_name.append("deltaiqr")
        temp_methods.append(METHODS[14])
        methods_dict.update({METHODS[14]: df})
        current_progress += 1

    if "deltaiqrmean" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[15]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=True, normalize=args.normalize, iqr_range=args.iqr_range)
        df = estimator.fit_predict(X=X, y=y)
        methods_save_name.append("deltaiqrmean")
        temp_methods.append(METHODS[15])
        methods_dict.update({METHODS[15]: df})
        current_progress += 1

    if "copa" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[16]), end="\r")
        estimator = COPA(q=args.q)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("copa")
        temp_methods.append(METHODS[16])
        methods_dict.update({METHODS[16]: df})
        current_progress += 1

    if "os" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[17]), end="\r")
        estimator = OutlierSumStatistic(q=args.q, iqr_range=args.iqr_range, two_sided_test=args.two_sided_test)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("os")
        temp_methods.append(METHODS[17])
        methods_dict.update({METHODS[17]: df})
        current_progress += 1

    if "ort" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[18]), end="\r")
        estimator = OutlierRobustTstatistic(q=args.q, iqr_range=args.iqr_range)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("ort")
        temp_methods.append(METHODS[18])
        methods_dict.update({METHODS[18]: df})
        current_progress += 1

    if "most" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[19]), end="\r")
        estimator = MOST(direction=args.direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("most")
        temp_methods.append(METHODS[19])
        methods_dict.update({METHODS[19]: df})
        current_progress += 1

    if "lsoss" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[20]), end="\r")
        estimator = LSOSS(direction=args.direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("lsoss")
        temp_methods.append(METHODS[20])
        methods_dict.update({METHODS[20]: df})
        current_progress += 1

    if "dids" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[21]), end="\r")
        estimator = DIDS(score_function=args.dids_scoref, direction=args.direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("dids")
        temp_methods.append(METHODS[21])
        methods_dict.update({METHODS[21]: df})
        current_progress += 1

    if "deco" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[22]), end="\r")
        if not os.path.exists(os.path.join(args.dspath, args.file_name + "_deco_features.csv")):
            raise Exception("Please provide DECO features file!")
        df = pd.read_csv(os.path.join(args.dspath, args.file_name + "_deco_features.csv"), sep=',')
        df = [(features_name[feature_ids[int(item[1][0])]], item[1][1]) for item in df.iterrows()]
        df = pd.DataFrame(df, columns=["features", "score"])
        methods_save_name.append("deco")
        temp_methods.append(METHODS[22])
        methods_dict.update({METHODS[22]: df})
        current_progress += 1

    if "phet_bd" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[23]), end="\r")
        phet_hvf_normalize = None
        if args.seurat_log_transform:
            phet_hvf_normalize = "log"
        estimator = PHeT(normalize=phet_hvf_normalize, iqr_range=args.iqr_range, num_subsamples=args.num_subsamples,
                         disp_type="hvf", calculate_deltadisp=args.calculate_deltadisp,
                         calculate_deltamean=args.calculate_deltamean, calculate_fisher=args.calculate_fisher,
                         calculate_disc_power=args.calculate_disc_power, bin_pvalues=args.bin_pvalues,
                         feature_weight=args.feature_weight, weight_range=args.weight_range)
        if args.exponentiate:
            df = estimator.fit_predict(X=np.exp(X), y=y, control_class=0, case_class=1)
        else:
            df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("phet_bd")
        temp_methods.append(METHODS[23])
        methods_dict.update({METHODS[23]: df})
        current_progress += 1

    if "phet_br" in args.methods:
        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[24]))
        estimator = PHeT(normalize="zscore", iqr_range=args.iqr_range, num_subsamples=args.num_subsamples,
                         disp_type="iqr", calculate_deltadisp=args.calculate_deltadisp,
                         calculate_deltamean=args.calculate_deltamean, calculate_fisher=args.calculate_fisher,
                         calculate_disc_power=args.calculate_disc_power, bin_pvalues=args.bin_pvalues,
                         feature_weight=args.feature_weight, weight_range=args.weight_range)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_save_name.append("phet_br")
        temp_methods.append(METHODS[24])
        methods_dict.update({METHODS[24]: df})

    ##########################################################################################################
    ######################                           EVALUATE                           ######################
    ##########################################################################################################
    if args.sort_by_pvalue:
        print('\n{0})- Sort features by the cut-off p-value at {1:.2f}...'.format(steps, args.alpha))
    else:
        print('\n{0})- Sort features by the score statistic...'.format(steps))
    steps = steps + 1

    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        if method_name in ['DECO', 't-statistic', 'Wilcoxon', 'LIMMA', 'KS']:
            continue
        if args.sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, alpha=args.alpha,
                                        X_map=None, map_genes=False, ttest=False)
        else:
            temp = sort_features(X=df, features_name=features_name, X_map=None,
                                 map_genes=False, ttest=False)
        methods_dict[method_name] = temp

    print("\t## Scoring results using up/down regulated features...")
    selected_regulated_features = args.top_k_features
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t\t>> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        if method_idx + 1 == len(temp_methods):
            print("\t\t\t--> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / len(temp_methods)) * 100,
                                                                        method_name))
        else:
            print("\t\t\t--> Progress: {0:.4f}%; Method: {1:30}".format((method_idx / len(temp_methods)) * 100,
                                                                        method_name), end="\r")
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        score = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                  metric=args.score_metric)
        list_scores.append(score)

    df = pd.DataFrame(list_scores, columns=["Scores"], index=temp_methods)
    df.to_csv(path_or_buf=os.path.join(args.rspath, args.file_name + "_features_scores.csv"), sep=",")
    print("\t## Plot bar plot using the top {0} features...".format(args.top_k_features))
    plot_barplot(X=list_scores, methods_name=temp_methods, metric=args.score_metric, suptitle=args.suptitle_name,
                 file_name=args.file_name, save_path=args.rspath)

    ##########################################################################################################
    ######################            Dimensionality Reduction & Clustering             ######################
    ##########################################################################################################
    print('\n{0})- Perform dimensionality reduction and clustering...'.format(steps))
    list_scores = list()
    print("\t## Plot UMAP using all features ({0})...".format(num_features))
    score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                      standardize=args.standardize, num_neighbors=args.num_neighbors, min_dist=args.min_dist,
                      perform_cluster=args.perform_cluster, cluster_type=args.cluster_type,
                      num_clusters=num_clusters, max_clusters=args.max_clusters, heatmap_plot=args.heatmap_plot,
                      num_jobs=args.num_jobs, suptitle=args.suptitle_name + "\nAll",
                      file_name=args.file_name + "_all", save_path=args.rspath)
    list_scores.append(score)

    if args.plot_top_k_features:
        print("\t## Plot UMAP using the top {0} features...".format(args.top_k_features))
    else:
        print("\t## Plot UMAP using the top predicted features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        save_name = methods_save_name[method_idx]
        if total_progress == method_idx + 1:
            print("\t\t>> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                     method_name))
        else:
            print("\t\t>> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                     method_name), end="\r")
        if args.plot_top_k_features:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:args.top_k_features]]
            temp_feature = [feature for idx, feature in enumerate(features_name) if
                            feature in df['features'].tolist()[:args.top_k_features]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
            temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        if num_features == 0:
            temp = [idx for idx, feature in enumerate(features_name)]
            temp_feature = [feature for idx, feature in enumerate(features_name)]
        scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
                           standardize=args.standardize, num_neighbors=args.num_neighbors, min_dist=args.min_dist,
                           perform_cluster=args.perform_cluster, cluster_type=args.cluster_type,
                           num_clusters=num_clusters, max_clusters=args.max_clusters, heatmap_plot=args.heatmap_plot,
                           num_jobs=args.num_jobs, suptitle=args.suptitle_name + "\n" + method_name,
                           file_name=args.file_name + "_" + save_name.lower(), save_path=args.rspath)
        df = pd.DataFrame(temp_feature, columns=["features"])
        df.to_csv(os.path.join(args.rspath, args.file_name + "_" + save_name.lower() + "_features.csv"),
                  sep=',', index=False, header=False)
        if args.export_spring:
            df = pd.DataFrame(X[:, temp])
            df.to_csv(
                path_or_buf=os.path.join(args.rspath, args.file_name + "_" + save_name.lower() + "_expression.csv"),
                sep=",", index=False, header=False)
        del df
        list_scores.append(scores)

    columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
               "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity",
               "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
    df = pd.DataFrame(list_scores, columns=columns, index=["All"] + temp_methods)
    df.to_csv(path_or_buf=os.path.join(args.rspath, args.file_name + "_cluster_quality.csv"), sep=",")

    print("\t## Plot bar plot using ARI metric...".format(args.top_k_features))
    plot_barplot(X=np.array(list_scores)[:, 12], methods_name=["All"] + temp_methods, metric="ari",
                 suptitle=args.suptitle_name, file_name=args.file_name, save_path=args.rspath)
