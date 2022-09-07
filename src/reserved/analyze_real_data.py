from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH
from utility.plot_utils import *
from utility.utils import sort_features


def train(num_jobs: int = 4):
    # Actions
    plot_results = False

    # Arguments
    top_k_features = 100
    map_genes = False
    direction = "both"
    calculate_pval = True
    num_iterations = 1000

    # Load expression data
    print("## Perform real experimental data analysis...")
    # Load chip data
    X_map = pd.read_csv(os.path.join(DATASET_PATH, "HU6800.chip"), sep='\t')
    X = pd.read_csv(os.path.join(DATASET_PATH, "leukemia_golub_two.csv"), sep=',')
    y = X["class"].to_numpy()
    features_name = X.drop(["class"], axis=1).columns.to_list()
    X = X.drop(["class"], axis=1).to_numpy()
    count = 1

    print("\t{0})- Unraveling cellular heterogeneity by analyzing intra-cellular variation (U-Het)...".format(count))
    estimator = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75),
                     calculate_pval=calculate_pval, num_iterations=num_iterations)
    df_uhet = estimator.fit_predict(X=X, y=y)
    df_uhet = sort_features(X=df_uhet, features_name=features_name, X_map=X_map,
                            map_genes=map_genes)
    count += 1

    # df_class = classifiers_results(X=X, y=y, features_name=features_name, num_features=num_features, standardize=True,
    #                                num_jobs=num_jobs, save_path=DATASET_PATH)
    # count += 1

    if np.unique(y).shape[0] == 2:
        print("\t{0})- The cancer outlier profile analysis (COPA)...".format(count))
        estimator = COPA(q=0.75, direction=direction, calculate_pval=calculate_pval, num_iterations=num_iterations)
        df_copa = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_copa = sort_features(X=df_copa, features_name=features_name, X_map=X_map, map_genes=map_genes)
        count += 1

        print("\t{0})- The outlier-sum statistic (OSS)...".format(count))
        estimator = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False,
                                        direction=direction, calculate_pval=calculate_pval,
                                        num_iterations=num_iterations)
        df_os = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_os = sort_features(X=df_os, features_name=features_name, X_map=X_map,
                              map_genes=map_genes)
        count += 1

        print("\t{0})- The outlier robust t-statistic (ORT)...".format(count))
        estimator = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75), direction=direction,
                                           calculate_pval=calculate_pval,
                                           num_iterations=num_iterations)
        df_ort = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_ort = sort_features(X=df_ort, features_name=features_name, X_map=X_map,
                               map_genes=map_genes)
        count += 1

        print("\t{0})- The maximum ordered subset t-statistics (MOST)...".format(count))
        estimator = MOST(direction=direction, calculate_pval=calculate_pval, num_iterations=num_iterations)
        df_most = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_most = sort_features(X=df_most, features_name=features_name, X_map=X_map,
                                map_genes=map_genes)
        count += 1

        print(
            "\t{0})- The least sum of ordered subset square t-statistic (LSOSS)...".format(count))
        estimator = LSOSS(direction=direction, calculate_pval=calculate_pval, num_iterations=num_iterations)
        df_lsoss = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_lsoss = sort_features(X=df_lsoss, features_name=features_name, X_map=X_map,
                                 map_genes=map_genes)
        count += 1

        print("\t{0})- The detection of imbalanced differential signal (DIDS)...".format(count))
        estimator = DIDS(score_function="quad", direction=direction, calculate_pval=calculate_pval,
                         num_iterations=num_iterations)
        df_dids = estimator.fit_predict(X=X, y=y, control_class=0, case_class=1)
        df_dids = sort_features(X=df_dids, features_name=features_name, X_map=X_map,
                                map_genes=map_genes)
        count += 1

        methods_df = [("COPA", df_copa), ("OS", df_os), ("ORT", df_ort), ("MOST", df_most),
                      ("LSOSS", df_lsoss), ("DIDS", df_dids), ("UHet", df_uhet)]
    else:
        methods_df = [("UHet", df_uhet)]

    if plot_results:
        print("\t{0})- Plot results using top k features...".format(count))
        # plot top k features
        for stat_name, df in methods_df:
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:top_k_features].tolist()]
            plot_umap(X=X[:, temp], y=y, num_features=top_k_features, standardize=True, num_jobs=num_jobs,
                      suptitle=stat_name.upper(), file_name=stat_name.lower(), save_path=DATASET_PATH)
            # plot_boxplot(X=X[:, temp], y=y, features_name=features_name[:top_k_features], num_features=top_k_features,
            #              standardize=True)
            # plot_clusters(X=X[:, temp], y=y, features_name=features_name[:top_k_features], num_features=top_k_features,
            #               standardize=True, cluster_type="spectral", num_clusters=0, num_neighbors=15, min_dist=0,
            #               heatmap=True, proportion=True, show_umap=True, num_jobs=num_jobs, suptitle=stat_name.upper(),
            #               file_name=stat_name.lower(), save_path=DATASET_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
