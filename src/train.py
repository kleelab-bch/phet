import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from model.copa import COPA
from model.deltaiqrmean import DeltaIQRMean
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.nonparametric_test import StudentTTest
from model.ort import OutlierRobustTstatistic
from model.oss import OutlierSumStatistic
from model.phet import PHeT
from utility.plot_utils import plot_umap, plot_barplot
from utility.utils import comparative_score
from utility.utils import sort_features, significant_features

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

METHODS = ["t-statistic", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "ΔIQR", "PHet"]


def train(args):
    # Setup the number of operations to employ
    steps = 1

    # Arguments
    topKfeatures = args.topKfeatures
    plot_topKfeatures = args.plot_topKfeatures
    if not args.sort_by_pvalue:
        plot_topKfeatures = True
    methods_save_name = ["ttest", "COPA", "OS", "ORT", "MOST", "LSOSS", "DIDS", "DeltaIQR"]
    if args.bin_KS_pvalues:
        methods_save_name.append("PHet_b")
    else:
        methods_save_name.append("PHet_nb")

    ##########################################################################################################
    ###################                  LOADING & PREPROCESSING DATASET                  ####################
    ##########################################################################################################
    print('\n{0})- Load and preprocess {1} dataset...'.format(steps, args.file_name))
    steps = steps + 1
    # Exprssion, classes, subtypes, donors, timepoints Files
    expression_file_name = args.file_name + "_matrix.mtx"
    features_file_name = args.file_name + "_feature_names.csv"
    markers_file = args.file_name + "_markers.csv"
    classes_file_name = args.file_name + "_classes.csv"
    subtypes_file = args.file_name + "_types.csv"
    differential_features_file = args.file_name + "_diff_features.csv"
    donors_file = args.file_name + "_donors.csv"
    timepoints_file = args.file_name + "_timepoints.csv"

    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(args.dspath, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))
    donors = []
    if os.path.exists(os.path.join(args.dspath, donors_file)):
        donors = pd.read_csv(os.path.join(args.dspath, donors_file), sep=',').dropna(axis=1)
        donors = [str(item[0]).lower() for item in donors.values.tolist()]
    timepoints = []
    if os.path.exists(os.path.join(args.dspath, timepoints_file)):
        timepoints = pd.read_csv(os.path.join(args.dspath, timepoints_file), sep=',').dropna(axis=1)
        timepoints = [str(item[0]).lower() for item in timepoints.values.tolist()]

    # Load features, expression, and class data
    features_name = pd.read_csv(os.path.join(args.dspath, features_file_name), sep=',')
    features_name = features_name["features"].to_list()
    y = pd.read_csv(os.path.join(args.dspath, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(args.dspath, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter data based on counts
    num_examples, num_features = X.shape
    if args.is_filter:
        example_sums = np.absolute(X).sum(1)
        examples_ids = np.where(example_sums > int(0.01 * num_features))[0]
        X = X[examples_ids]
        y = y[examples_ids]
        subtypes = np.array(subtypes)[examples_ids].tolist()
        if len(donors) != 0:
            donors = np.array(donors)[examples_ids].tolist()
        if len(timepoints) != 0:
            timepoints = np.array(timepoints)[examples_ids].tolist()
        num_examples, num_features = X.shape
        del example_sums, examples_ids
        temp = np.absolute(X)
        temp = (temp * 1e6) / temp.sum(axis=1).reshape((num_examples, 1))
        temp[temp > 1] = 1
        temp[temp != 1] = 0
        feature_sums = temp.sum(0)
        del temp
        feature_ids = np.where(feature_sums > int(0.01 * num_examples))[0]
        features_name = np.array(features_name)[feature_ids].tolist()
        X = X[:, feature_ids]
        num_examples, num_features = X.shape
        del feature_sums

    # Save subtypes for SPRING
    if args.export_spring:
        groups = []
        groups.append(["subtypes"] + subtypes)
        if len(donors) != 0:
            groups.append(["donors"] + donors)
        if len(timepoints) != 0:
            groups.append(["timepoints"] + timepoints)
        df = pd.DataFrame(groups)
        df.to_csv(os.path.join(args.rspath, args.file_name + "_groups.csv"), sep=',',
                  index=False, header=False)
        del df

    # Load up/down regulated features
    top_features_true = -1
    if os.path.exists(os.path.join(args.dspath, markers_file)):
        top_features_true = pd.read_csv(os.path.join(args.dspath, markers_file)).replace(np.nan, -1)
        top_features_true = list(set([item for item in top_features_true.to_numpy().flatten() if item != -1]))
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]
        topKfeatures = sum(top_features_true)
    elif os.path.exists(os.path.join(args.dspath, differential_features_file)):
        top_features_true = pd.read_csv(os.path.join(args.dspath, differential_features_file), sep=',',
                                        index_col="ID")
        temp = [feature for feature in top_features_true.index.to_list() if str(feature) in features_name]
        if top_features_true.shape[1] > 0:
            top_features_true = top_features_true.loc[temp]
            temp = top_features_true[top_features_true["adj.P.Val"] <= args.pvalue]
            if temp.shape[0] < topKfeatures:
                temp = top_features_true[:topKfeatures - 1]
                if args.sort_by_pvalue and temp.shape[0] == 0:
                    plot_topKfeatures = True
            top_features_true = [str(feature_idx) for feature_idx in temp.index.to_list()[:topKfeatures]]
        else:
            top_features_true = temp
            topKfeatures = len(top_features_true)
        top_features_true = [1 if feature in top_features_true else 0 for idx, feature in enumerate(features_name)]

    ##########################################################################################################
    #####################                            Inference                            ####################
    ##########################################################################################################
    print('\n{0})- Infer subtypes using {1} dataset...'.format(steps, args.file_name))
    print("\t >> Sample size: {0}; Feature size: {1}; Subtype size: {2}".format(X.shape[0], X.shape[1],
                                                                                len(np.unique(subtypes))))
    steps = steps + 1
    current_progress = 1
    total_progress = len(METHODS)
    methods_dict = dict()

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[0]), end="\r")
    estimator = StudentTTest(direction=args.direction)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[0]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[1]), end="\r")
    estimator = COPA(q=args.q)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[1]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[2]), end="\r")
    estimator = OutlierSumStatistic(q=args.q, iqr_range=args.iqr_range, two_sided_test=args.direction)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[2]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[3]), end="\r")
    estimator = OutlierRobustTstatistic(q=args.q, iqr_range=args.iqr_range)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[3]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[4]), end="\r")
    estimator = MOST()
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[4]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[5]), end="\r")
    estimator = LSOSS(direction=args.direction)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[5]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[6]), end="\r")
    estimator = DIDS(score_function=args.dids_scoref, direction=args.direction)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[6]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[7]), end="\r")
    estimator = DeltaIQRMean(normalize=args.normalize, iqr_range=args.iqr_range)
    df = estimator.fit_predict(X=X, y=y)
    methods_dict.update({METHODS[7]: df})
    current_progress += 1

    print("\t >> Progress: {0:.4f}%; Method: {1:20}".format((current_progress / total_progress) * 100,
                                                            METHODS[8]))
    estimator = PHeT(normalize=args.normalize, iqr_range=args.iqr_range, num_subsamples=args.num_subsamples,
                     calculate_deltaiqr=args.calculate_deltaiqr, calculate_fisher=args.calculate_fisher,
                     calculate_profile=args.calculate_profile, bin_pvalues=args.bin_KS_pvalues,
                     feature_weight=args.feature_weight, weight_range=args.weight_range)
    df = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
    methods_dict.update({METHODS[8]: df})

    ##########################################################################################################
    ######################                           EVALUATE                           ######################
    ##########################################################################################################
    if args.sort_by_pvalue:
        print('\n{0})- Sort features by the cut-off {1:.2f} p-value...'.format(steps, args.pvalue))
    else:
        print('\n{0})- Sort features by the score statistic...'.format(steps))
    steps = steps + 1
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if args.sort_by_pvalue:
            temp = significant_features(X=df, features_name=features_name, alpha=args.pvalue)
        else:
            temp = sort_features(X=df, features_name=features_name)
        methods_dict[method_name] = temp
    del df

    if top_features_true != -1:
        print("\t >> Scoring results using known regulated features...")
        selected_regulated_features = topKfeatures
        temp = np.sum(top_features_true)
        if selected_regulated_features > temp:
            selected_regulated_features = temp
        print("\t >> Number of up/down regulated features: {0}".format(selected_regulated_features))
        list_scores = list()
        for method_idx, item in enumerate(methods_dict.items()):
            if method_idx + 1 == len(METHODS):
                print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / len(METHODS)) * 100,
                                                                          METHODS[method_idx]))
            else:
                print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format((method_idx / len(METHODS)) * 100,
                                                                          METHODS[method_idx]), end="\r")
            method_name, df = item
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:selected_regulated_features].tolist()]
            top_features_pred = np.zeros((len(top_features_true)))
            top_features_pred[temp] = 1
            score = comparative_score(pred_features=top_features_pred,
                                      true_features=top_features_true,
                                      metric="f1")
            list_scores.append(score)

        df = pd.DataFrame(list_scores, columns=["Scores"], index=METHODS)
        df.to_csv(path_or_buf=os.path.join(args.rspath, args.file_name + "_features_scores.csv"), sep=",")
        print("\t >> Visualize barplot using the top {0} features...".format(topKfeatures))
        plot_barplot(X=list_scores, methods_name=METHODS, metric="f1", suptitle=args.suptitle_name,
                     file_name=args.file_name, save_path=args.rspath)

    ##########################################################################################################
    ######################            Dimensionality Reduction & Clustering             ######################
    ##########################################################################################################
    print('\n{0})- Perform dimensionality reduction and clustering...'.format(steps))
    steps = steps + 1
    temp = np.copy(y)
    temp = temp.astype(str)
    temp[np.where(y == 0)[0]] = args.control_name
    temp[np.where(y == 1)[0]] = args.case_name
    y = temp
    list_scores = list()
    score = 0
    print("\t >> Visualize UMAP using all features ({0})...".format(num_features))
    score = plot_umap(X=X, y=y, subtypes=subtypes, features_name=features_name, num_features=num_features,
                      standardize=args.standardize, num_neighbors=args.num_neighbors, min_dist=args.min_dist,
                      perform_cluster=args.perform_cluster, cluster_type=args.cluster_type, num_clusters=num_clusters,
                      max_clusters=args.max_clusters, heatmap_plot=args.heatmap_plot, num_jobs=args.num_jobs,
                      suptitle=args.suptitle_name + "\nAll", file_name=args.file_name + "_all", save_path=args.rspath)
    list_scores.append(score)
    if top_features_true != -1:
        print("\t >> Visualize UMAP using marker features ({0})...".format(sum(top_features_true)))
        temp = np.where(np.array(top_features_true) == 1)[0]
        score = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=features_name, num_features=temp.shape[0],
                          standardize=args.standardize, num_neighbors=args.num_neighbors, min_dist=args.min_dist,
                          perform_cluster=args.perform_cluster, cluster_type=args.cluster_type,
                          num_clusters=num_clusters, max_clusters=args.max_clusters, heatmap_plot=args.heatmap_plot,
                          num_jobs=args.num_jobs, suptitle=args.suptitle_name + "\nMarkers",
                          file_name=args.file_name + "_markers", save_path=args.rspath)
        list_scores.append(score)

    if plot_topKfeatures:
        print("\t >> Visualize UMAP using the top {0} features...".format(topKfeatures))
    else:
        print("\t >> Visualize UMAP using the top features for each method...")
    for method_idx, item in enumerate(methods_dict.items()):
        method_name, df = item
        method_name = METHODS[method_idx]
        save_name = methods_save_name[method_idx]
        if total_progress == method_idx + 1:
            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                      method_name))
        else:
            print("\t\t--> Progress: {0:.4f}%; Method: {1:20}".format(((method_idx + 1) / total_progress) * 100,
                                                                      method_name), end="\r")
        if plot_topKfeatures:
            temp = [idx for idx, feature in enumerate(features_name) if
                    feature in df['features'].tolist()[:topKfeatures]]
            temp_feature = [feature for idx, feature in enumerate(features_name) if
                            feature in df['features'].tolist()[:topKfeatures]]
        else:
            temp = [idx for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
            temp_feature = [feature for idx, feature in enumerate(features_name) if feature in df['features'].tolist()]
        num_features = len(temp)
        score = plot_umap(X=X[:, temp], y=y, subtypes=subtypes, features_name=temp_feature, num_features=num_features,
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
        list_scores.append(score)
    index = ["All"]
    if top_features_true != -1:
        index += ["Markers"]
    df = pd.DataFrame(list_scores, columns=["Scores"], index=index + METHODS)
    df.to_csv(path_or_buf=os.path.join(args.rspath, args.file_name + "_cluster_quality.csv"), sep=",")

    print("\t >> Visualize barplot using to demonstrate clustering accuracy...".format(topKfeatures))
    plot_barplot(X=list_scores, methods_name=index + METHODS, metric="ari",
                 suptitle=args.suptitle_name, file_name=args.file_name,
                 save_path=args.rspath)
