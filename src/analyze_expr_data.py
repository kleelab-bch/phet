import os

import numpy as np
import pandas as pd
import seaborn as sns

from model.cleanse import CLEANSE
from model.cleanse_hierarchical import HCLEANSE
from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH, RESULT_PATH
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()


def train(num_jobs: int = 4):
    # Arguments
    direction = "both"
    topKfeatures = 100
    minimum_topfeatures = 50
    pvalue = 0.01
    sort_by_pvalue = True
    plot_topKfeatures = False

    # 1. Microarray datasets: tnbc, pdac, colon, leukemia_golub, lung, bcca1, myelodysplastic_mds1, and myelodysplastic_mds2
    # 2. scRNA datasets: camp2, darmanis, lake, yan, camp1, baron, segerstolpe, wang, li, and patel
    file_name = "myelodysplastic_mds1"
    expression_file_name = file_name + "_matrix"
    regulated_features_file = file_name + "_features"
    subtypes_file = file_name + "_types"

    # Load expression data
    X = pd.read_csv(os.path.join(DATASET_PATH, expression_file_name + ".csv"), sep=',').dropna(axis=1)
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    # Load up/down regulated features
    top_features_true = pd.read_csv(os.path.join(DATASET_PATH, regulated_features_file + ".csv"), sep=',')
    temp = top_features_true[top_features_true["adj.P.Val"] <= pvalue]
    if len(temp["ID"]) < minimum_topfeatures:
        temp = top_features_true.loc[:topKfeatures-1]
    top_features_true = [str(feature_idx) for feature_idx in temp["ID"].to_list()[:topKfeatures]]
    top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file + ".csv"), sep=',').dropna(axis=1)
    subtypes = subtypes["subtypes"].to_list()
    
    print("## Perform experimental studies using {0} data...".format(expression_file_name))
    current_progress = 1
    total_progress = 10

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "COPA"), end="\r")
    estimator = COPA(q=0.75, direction=direction, calculate_pval=False)
    df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "OS"), end="\r")
    estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False, direction=direction,
                                    calculate_pval=False)
    df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "ORT"), end="\r")
    estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                       calculate_pval=False)
    df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "MOST"), end="\r")
    estimator = MOST(direction=direction, calculate_pval=False)
    df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "LSOSS"), end="\r")
    estimator = LSOSS(direction=direction, calculate_pval=False)
    df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "DIDS"), end="\r")
    estimator = DIDS(score_function="quad", direction=direction, calculate_pval=False)
    df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (zscore)"), end="\r")
    estimator = UHeT(normalize="zscore", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_z = estimator.fit_predict(X=X, y=y)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            "U-Het (robust)"), end="\r")
    estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75), calculate_pval=False)
    df_uhet_r = estimator.fit_predict(X=X, y=y)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "CLEANSE (zscore)"), end="\r")
    estimator = CLEANSE(normalize="zscore", q=0.75, iqr_range=(25, 75), num_subsamples=100, subsampling_size=None,
                        significant_p=0.05, partition_by_anova=False, num_components=10, num_subclusters=10,
                        binary_clustering=True, calculate_pval=False, num_rounds=50, num_jobs=num_jobs)
    df_cleanse_z = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "CLEANSE (robust)"))
    estimator = CLEANSE(normalize="robust", q=0.75, iqr_range=(25, 75), num_subsamples=100, subsampling_size=None,
                        significant_p=0.05, partition_by_anova=False, num_components=10, num_subclusters=10,
                        binary_clustering=True, calculate_pval=False, num_rounds=50, num_jobs=num_jobs)
    df_cleanse_r = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    # df_cleanse = estimator.fit_predict(X=X, y=None, control_class=0, case_class=1)
    # df_cleanse = estimator.fit_predict(X=X, y=None, partition_data=True, control_class=0, case_class=1)
    # current_progress += 1

    methods_df = dict({"COPA": df_copa, "OS": df_os, "ORT": df_ort, "MOST": df_most,
                       "LSOSS": df_lsoss, "DIDS": df_dids, "UHet_zscore": df_uhet_z,
                       "UHet_robust": df_uhet_r, "CLEANSE_zscore": df_cleanse_z,
                       "CLEANSE_robust": df_cleanse_r})

    # print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100, "HCLEANSE"))
    # estimator = HCLEANSE(normalize="robust", q=0.75, iqr_range=(25, 75), num_subsamples=10, subsampling_size=None,
    #                      significant_p=0.05, partition_by_anova=False, num_components=10, num_subclusters=10,
    #                      binary_clustering=False, feature_weight=[0.4, 0.3, 0.2, 0.1], max_features=100,
    #                      max_depth=3, min_samples_split=5, num_estimators=5, num_rounds=50, calculate_pval=False,
    #                      num_jobs=num_jobs)
    # df_hcleanse = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1, return_best_features=True,
    #                                    return_clusters=False)
    # df_hcleanse = df_hcleanse[0]

    # methods_df = dict({"COPA": df_copa, "OS": df_os, "ORT": df_ort, "MOST": df_most,
    #                    "LSOSS": df_lsoss, "DIDS": df_dids, "UHet_zscore": df_uhet_z,
    #                    "UHet_robust": df_uhet_r, "CLEANSE": df_cleanse, "HCLEANSE": df_hcleanse})

    if sort_by_pvalue:
        print("## Sort features by the cut-off {0:.2f} p-value...".format(pvalue))
        for stat_name, df in methods_df.items():
            temp = significant_features(X=df, features_name=features_name, pvalue=pvalue,
                                        X_map=None, map_genes=False, ttest=False)
            methods_df[stat_name] = temp
    else:
        print("## Sort features by the score statistic...".format())
        for stat_name, df in methods_df.items():
            temp = sort_features(X=df, features_name=features_name, X_map=None, map_genes=False, ttest=False)
            methods_df[stat_name] = temp
    del df_copa, df_os, df_ort, df_most, df_lsoss, df_dids, df_uhet_z, df_uhet_r, df_cleanse_r

    print("## Scoring results using known regulated features...")
    selected_regulated_features = topKfeatures
    temp = np.sum(top_features_true)
    if selected_regulated_features > temp:
        selected_regulated_features = temp
    print("\t\t>> Number of up/down regulated features: {0}".format(selected_regulated_features))
    list_scores = list()
    for stat_name, df in methods_df.items():
        temp = [idx for idx, feature in enumerate(features_name)
                if feature in df['features'][:selected_regulated_features].tolist()]
        top_features_pred = np.zeros((len(top_features_true)))
        top_features_pred[temp] = 1
        score = comparative_score(top_features_pred=top_features_pred, top_features_true=top_features_true)
        list_scores.append(score)

    print("## Plot barplot using the top {0} features...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=list(methods_df.keys()), file_name=expression_file_name,
                 save_path=RESULT_PATH)

    if plot_topKfeatures:
        print("## Plot results using the top {0} features...".format(topKfeatures))
    else:
        print("## Plot results using the top features for each method...")
    for method_idx, item in enumerate(methods_df.items()):
        stat_name, df = item
        if total_progress == method_idx + 1:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    stat_name))
        else:
            print("\t >> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                    stat_name), end="\r")
        if plot_topKfeatures:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:topKfeatures]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        plot_umap(X=X[:, temp], y=subtypes, num_features=num_features, standardize=True, num_jobs=num_jobs,
                  suptitle=stat_name.upper(), file_name=expression_file_name + "_" + stat_name.lower(),
                  save_path=RESULT_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=8)