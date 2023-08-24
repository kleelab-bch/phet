import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot


sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "t-statistic+Gamma", "Wilcoxon", "Wilcoxon+Gamma",
           "KS", "KS+Gamma", "LIMMA", "LIMMA+Gamma", "Dispersion (composite)",
           "Dispersion (by condition)", "ΔDispersion", "ΔDispersion+ΔMean",
           "IQR (composite)", "IQR (by condition)", "ΔIQR", "ΔIQR+ΔMean",
           "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DECO",
           "PHet (ΔDispersion)", "PHet"]


def train(num_jobs: int = 4):
    # Filtering arguments
    minimum_samples = 5
    # Models parameters
    methods_save_name = ["ttest_p", "ttest_g", "wilcoxon_p", "wilcoxon_g", "ks_p", "ks_g",
                         "limma_p", "limma_g", "dispersion_a", "dispersion_c", "deltadispersion",
                         "deltadispersionmean", "iqr_a", "iqr_c", "deltaiqr", "deltaiqrmean",
                         "copa", "os", "ort", "most", "lsoss", "dids", "deco", "phet_bd", "phet_br"]
    # Clustering and UMAP parameters
    export_spring = False
    top_k_features = 100
    num_neighbors = 5
    max_clusters = 10

    # Descriptions of the data
    # datasets = ["bc_ccgse3726", "bladdergse89", "braintumor", "glioblastoma", "leukemia_golub", "lung"]
    # suptitle_names = ["GSE3726", "GSE89", "Braintumor", "Glioblastoma", "Leukemia", "Lung"]
    datasets = ["baron1"]
    suptitle_names = ["Baron"]
    cluster_type = "spectral"

    # datasets = ["allgse412", "gastricgse2685", "lunggse1987", "mll", "srbct"]
    # suptitle_names = ["GSE412", "GSE2685", "GSE1987", "MLL", "SRBCT"]
    # datasets = ["camp1", "darmanis", "li", "patel", "yan"]
    # suptitle_names = ["Camp", "Darmanis", "Li", "Patel", "Yan"]
    # cluster_type = "kmeans"

    standardize = False
    data_folder = "scRNA" # "scRNA" "microarray"
    if data_folder == "microarray":
        standardize = False

    for data_idx, data_name in enumerate(datasets):
        suptitle_name = suptitle_names[data_idx]
        # Expression, classes, subtypes, donors, timepoints files
        expression_file_name = data_name + "_matrix.mtx"
        features_file_name = data_name + "_feature_names.csv"
        classes_file_name = data_name + "_classes.csv"
        subtypes_file = data_name + "_types.csv"

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

        print("{0}/{1}) Perform experimental studies using {2} data...".format(data_idx + 1, len(datasets),
                                                                               suptitle_name))
        print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                    len(np.unique(subtypes))))
        total_progress = len(METHODS)

        list_scores = list()
        print("   ## Plot UMAP using all features ({0})...".format(num_features))
        score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                          standardize=standardize, num_neighbors=num_neighbors, min_dist=0, perform_cluster=True,
                          cluster_type=cluster_type, num_clusters=num_clusters, max_clusters=max_clusters,
                          heatmap_plot=False, num_jobs=num_jobs, suptitle=suptitle_name + "\nAll",
                          file_name=data_name + "_all", save_path=RESULT_PATH)
        list_scores.append(score)

        for method_idx, save_name in enumerate(methods_save_name):
            method_name = METHODS[method_idx]
            df = pd.read_csv(os.path.join(RESULT_PATH, data_folder, 
                                          data_name + "_" + save_name + "_features.csv"),
                                 sep=',', header=None)
            if total_progress == method_idx + 1:
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                        method_name))
            else:
                print("\t >> Progress: {0:.4f}%; Method: {1:30}".format(((method_idx + 1) / total_progress) * 100,
                                                                        method_name), end="\r")
            temp = [idx for idx, feature in enumerate(features_name) if feature in df[0].tolist()]
            temp_feature = [feature for idx, feature in enumerate(features_name) if
                                feature in df[0].tolist()]
            num_features = len(temp)
            if num_features == 0:
                temp = [idx for idx, feature in enumerate(features_name)]
                temp_feature = [feature for idx, feature in enumerate(features_name)]
            scores = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature,
                               num_features=num_features,
                               standardize=standardize, num_neighbors=num_neighbors, min_dist=0.0, perform_cluster=True,
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
