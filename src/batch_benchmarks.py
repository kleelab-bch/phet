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
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma",
           "KS", "KS+Gamma", "LIMMA", "LIMMA+Gamma", "HVF (composite)",
           "HVF (by condition)", "ΔHVF", "ΔHVF+ΔMean", "IQR (composite)",
           "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean", "COPA", "OS", "ORT",
           "MOST", "LSOSS", "DIDS", "DECO", "PHet (ΔHVF)", "PHet"]


def train(num_jobs: int = 4):
    # Filtering arguments
    minimum_samples = 5
    pvalue = 0.01

    # Models parameters
    direction = "both"
    log_transform = False
    phet_hvf_normalize = None
    if log_transform:
        phet_hvf_normalize = "log"
    methods_save_name = ["ttest_p", "ttest_g", "wilcoxon_p", "wilcoxon_g", "ks_p", "ks_g", "limma_p",
                         "limma_g", "hvf_a", "hvf_c", "deltahvf", "deltahvfmean", "iqr_a", "iqr_c",
                         "deltaiqr", "deltaiqrmean", "copa", "os", "ort", "most", "lsoss", "dids",
                         "deco", "phet_bh", "phet_br"]
    # Clustering and UMAP parameters
    sort_by_pvalue = True
    export_spring = False
    top_k_features = 100
    plot_top_k_features = False
    if not sort_by_pvalue:
        plot_top_k_features = True
    num_neighbors = 5
    max_clusters = 10
    feature_metric = "f1"

    # Descriptions of the data
    # datasets = ["bc_ccgse3726", "bladdergse89", "braintumor", "glioblastoma", "leukemia_golub", "lung"]
    # suptitle_names = ["GSE3726", "GSE89", "Braintumor", "Glioblastoma", "Leukemia", "Lung"]
    # datasets = ["baron1"]
    # suptitle_names = ["Baron"]
    # cluster_type = "spectral"

    # datasets = ["allgse412", "gastricgse2685", "lunggse1987", "mll", "srbct"]
    # suptitle_names = ["GSE412", "GSE2685", "GSE1987", "MLL", "SRBCT"]
    datasets = ["camp1", "darmanis", "li", "patel", "yan"]
    suptitle_names = ["Camp", "Darmanis", "Li", "Patel", "Yan"]
    cluster_type = "kmeans"

    for data_idx, data_name in enumerate(datasets):

        suptitle_name = suptitle_names[data_idx]
        # Expression, classes, subtypes, donors, timepoints files
        expression_file_name = data_name + "_matrix.mtx"
        features_file_name = data_name + "_feature_names.csv"
        classes_file_name = data_name + "_classes.csv"
        subtypes_file = data_name + "_types.csv"
        differential_features_file = data_name + "_limma_features.csv"

        # Load subtypes file
        subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
        subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
        num_clusters = len(np.unique(subtypes))

        # Load features, expression, and class data
        features_name = pd.read_csv(os.path.join(DATASET_PATH, features_file_name), sep=',')
        features_name = features_name["features"].to_list()
        y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
        y = y["classes"].to_numpy()
        X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
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
        if export_spring:
            df = pd.DataFrame(subtypes, columns=["subtypes"]).T
            df.to_csv(os.path.join(RESULT_PATH, data_name + "_subtypes.csv"), sep=',', header=False)
            del df

        # Load up/down regulated features
        top_features_true = pd.read_csv(os.path.join(DATASET_PATH, differential_features_file), sep=',',
                                        index_col="ID")
        temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
        if top_features_true.shape[1] > 0:
            top_features_true = top_features_true.loc[temp]
            temp = top_features_true[top_features_true["adj.P.Val"] <= pvalue]
            if temp.shape[0] < top_k_features:
                temp = top_features_true[:top_k_features - 1]
                if sort_by_pvalue and temp.shape[0] == 0:
                    plot_top_k_features = True
            top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:top_k_features]]
        else:
            top_features_true = temp
            top_k_features = len(top_features_true)
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

        print("{0}/{1}) Perform experimental studies using {2} data...".format(data_idx + 1, len(datasets),
                                                                               suptitle_name))
        print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                    len(np.unique(subtypes))))
        current_progress = 1
        total_progress = len(METHODS)
        methods_dict = dict()

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[0]), end="\r")
        estimator = StudentTTest(use_statistics=False, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] <= pvalue]
        methods_dict.update({METHODS[0]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[1]), end="\r")
        estimator = StudentTTest(use_statistics=True, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[1]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[2]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=False, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] <= pvalue]
        methods_dict.update({METHODS[2]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[3]), end="\r")
        estimator = WilcoxonRankSumTest(use_statistics=True, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[3]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[4]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=False, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False,
                           ttest=False, ascending=True)
        df = df[df["score"] <= pvalue]
        methods_dict.update({METHODS[4]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[5]), end="\r")
        estimator = KolmogorovSmirnovTest(use_statistics=True, direction=direction, adjust_pvalue=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[5]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[6]), end="\r")
        df = pd.read_csv(os.path.join(DATASET_PATH, data_name + "_limma_features.csv"), sep=',')
        df = df[["ID", "adj.P.Val", "B"]]
        df = df[df["adj.P.Val"] <= pvalue]
        df = df[["ID", "B"]]
        df.columns = ["features", "score"]
        methods_dict.update({METHODS[6]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[7]), end="\r")
        df = pd.read_csv(os.path.join(DATASET_PATH, data_name + "_limma_features.csv"), sep=',')
        df = df[["ID", "B"]]
        temp = [features_name.index(item) for item in df["ID"].to_list() if item in features_name]
        df = np.absolute(df.iloc[temp]["B"].to_numpy()[:, None])
        methods_dict.update({METHODS[7]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[8]), end="\r")
        estimator = SeuratHVF(per_condition=False, log_transform=log_transform,
                              num_top_features=num_features, min_disp=0.5,
                              min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_dict.update({METHODS[8]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[9]), end="\r")
        estimator = SeuratHVF(per_condition=True, log_transform=log_transform,
                              num_top_features=num_features, min_disp=0.5,
                              min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_dict.update({METHODS[9]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[10]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=False, log_transform=log_transform,
                                 num_top_features=num_features, min_disp=0.5,
                                 min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_dict.update({METHODS[10]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[11]), end="\r")
        estimator = DeltaHVFMean(calculate_deltamean=True, log_transform=log_transform,
                                 num_top_features=num_features, min_disp=0.5,
                                 min_mean=0.0125, max_mean=3)
        temp_X = deepcopy(X)
        df = estimator.fit_predict(X=temp_X, y=y)
        del temp_X
        methods_dict.update({METHODS[11]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[12]), end="\r")
        estimator = HIQR(per_condition=False, normalize="zscore", iqr_range=(25, 75))
        df = estimator.fit_predict(X=X, y=y)
        methods_dict.update({METHODS[12]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[13]), end="\r")
        estimator = HIQR(per_condition=True, normalize="zscore", iqr_range=(25, 75))
        df = estimator.fit_predict(X=X, y=y)
        methods_dict.update({METHODS[13]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[14]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=False, normalize="zscore", iqr_range=(25, 75))
        df = estimator.fit_predict(X=X, y=y)
        methods_dict.update({METHODS[14]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[15]), end="\r")
        estimator = DeltaIQRMean(calculate_deltamean=True, normalize="zscore", iqr_range=(25, 75))
        df = estimator.fit_predict(X=X, y=y)
        methods_dict.update({METHODS[15]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[16]), end="\r")
        estimator = COPA(q=75)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[16]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[17]), end="\r")
        estimator = OutlierSumStatistic(q=75, iqr_range=(25, 75), two_sided_test=False)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[17]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[18]), end="\r")
        estimator = OutlierRobustTstatistic(q=75, iqr_range=(25, 75))
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[18]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[19]), end="\r")
        estimator = MOST(direction=direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[19]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[20]), end="\r")
        estimator = LSOSS(direction=direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[20]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[21]), end="\r")
        estimator = DIDS(score_function="tanh", direction=direction)
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[21]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[22]), end="\r")
        df = pd.read_csv(os.path.join(DATASET_PATH, data_name + "_deco_features.csv"), sep=',')
        df = [(features_name[feature_ids[int(item[1][0])]], item[1][1]) for item in df.iterrows()]
        df = pd.DataFrame(df, columns=["features", "score"])
        methods_dict.update({METHODS[22]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[23]), end="\r")
        estimator = PHeT(normalize=phet_hvf_normalize, iqr_range=(25, 75), num_subsamples=1000, delta_type="hvf",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[23]: df})
        current_progress += 1

        print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((current_progress / total_progress) * 100,
                                                                METHODS[24]))
        estimator = PHeT(normalize="zscore", iqr_range=(25, 75), num_subsamples=1000, delta_type="iqr",
                         calculate_deltadisp=True, calculate_deltamean=False, calculate_fisher=True,
                         calculate_profile=True, bin_pvalues=True, feature_weight=[0.4, 0.3, 0.2, 0.1],
                         weight_range=[0.2, 0.4, 0.8])
        df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        methods_dict.update({METHODS[24]: df})

        if sort_by_pvalue:
            print("   ## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
        else:
            print("   ## Sort features by the score statistic...".format())
        for method_idx, item in enumerate(methods_dict.items()):
            method_name, df = item
            method_name = METHODS[method_idx]
            save_name = methods_save_name[method_idx]
            if method_name in ['DECO', 't-statistic', 'Wilcoxon', 'LIMMA', 'KS']:
                continue
            if sort_by_pvalue:
                temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                            X_map=None, map_genes=False, ttest=False)
            else:
                temp = sort_features(X=df, features_name=features_name, X_map=None,
                                     map_genes=False, ttest=False)
            methods_dict[method_name] = temp

        print("   ## Scoring results using up/down regulated features...")
        selected_regulated_features = top_k_features
        temp = np.sum(top_features_true)
        if selected_regulated_features > temp:
            selected_regulated_features = temp
        print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
        list_scores = list()
        for method_idx, item in enumerate(methods_dict.items()):
            if method_idx + 1 == len(METHODS):
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / len(METHODS)) * 100,
                                                                        METHODS[method_idx]))
            else:
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format((method_idx / len(METHODS)) * 100,
                                                                        METHODS[method_idx]), end="\r")
            method_name, df = item
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:selected_regulated_features].tolist()]
            top_features_pred = np.zeros((len(top_features_true)))
            top_features_pred[temp] = 1
            score = comparative_score(pred_features=top_features_pred, true_features=top_features_true,
                                      metric=feature_metric)
            list_scores.append(score)

        df = pd.DataFrame(list_scores, columns=["Scores"], index=METHODS)
        df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_features_scores.csv"), sep=",")
        print("   ## Plot bar plot using the top {0} features...".format(top_k_features))
        plot_barplot(X=list_scores, methods_name=METHODS, metric=feature_metric, suptitle=suptitle_name,
                     file_name=data_name, save_path=RESULT_PATH)

        list_scores = list()
        print("   ## Plot UMAP using all features ({0})...".format(num_features))
        score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                          standardize=True, num_neighbors=num_neighbors, min_dist=0, perform_cluster=True,
                          cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                          heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\nAll",
                          file_name=data_name + "_all", save_path=RESULT_PATH)
        list_scores.append(score)

        if plot_top_k_features:
            print("   ## Plot UMAP using the top {0} features...".format(top_k_features))
        else:
            print("   ## Plot UMAP using the top features for each method...")
        for method_idx, item in enumerate(methods_dict.items()):
            method_name, df = item
            method_name = METHODS[method_idx]
            save_name = methods_save_name[method_idx]
            if total_progress == method_idx + 1:
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                        method_name))
            else:
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                        method_name), end="\r")
            if plot_top_k_features:
                temp = [idx for idx, feature in enumerate(features_name) if
                        feature in df['features'].tolist()[:top_k_features]]
                temp_feature = [feature for idx, feature in enumerate(features_name) if
                                feature in df['features'].tolist()[:top_k_features]]
            else:
                temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
                temp_feature = [feature for idx, feature in enumerate(features_name) if
                                feature in df['features'].tolist()]
            num_features = len(temp)
            scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature,
                               num_features=num_features,
                               standardize=True, num_neighbors=num_neighbors, min_dist=0.0, perform_cluster=True,
                               cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                               heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\n" + method_name,
                               file_name=data_name + "_" + save_name.lower(), save_path=RESULT_PATH)
            df = pd.DataFrame(temp_feature, columns=["features"])
            df.to_csv(os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_features.csv"),
                      sep=',', index=False, header=False)
            if export_spring:
                df = pd.DataFrame(X[:, temp])
                df.to_csv(
                    path_or_buf=os.path.join(RESULT_PATH, data_name + "_" + save_name.lower() + "_expression.csv"),
                    sep=",", index=False, header=False)
            del df
            list_scores.append(scores)

        columns = ["Complete Diameter Distance", "Average Diameter Distance", "Centroid Diameter Distance",
                   "Single Linkage Distance", "Maximum Linkage Distance", "Average Linkage Distance",
                   "Centroid Linkage Distance", "Ward's Distance", "Silhouette", "Homogeneity",
                   "Completeness", "V-measure", "Adjusted Rand Index", "Adjusted Mutual Info"]
        df = pd.DataFrame(list_scores, columns=columns, index=["All"] + METHODS)
        df.to_csv(path_or_buf=os.path.join(RESULT_PATH, data_name + "_cluster_quality.csv"), sep=",")

        print("   ## Plot bar plot using ARI metric...\n".format(top_k_features))
        plot_barplot(X=np.array(list_scores)[:, 12], methods_name=["All"] + METHODS, metric="ari",
                     suptitle=suptitle_name, file_name=data_name, save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=10)
