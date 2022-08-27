from sklearn.metrics import jaccard_score

from model.copa import COPA
from model.dids import DIDS
from model.lsoss import LSOSS
from model.most import MOST
from model.ors import OutlierRobustStatistic
from model.oss import OutlierSumStatistic
from model.uhet import UHeT
from utility.file_path import DATASET_PATH
from utility.plot_utils import *


def sort_features(X, features_name, X_map=None, map_genes: bool = True, ttest: bool = False):
    df = pd.concat([pd.DataFrame(features_name), pd.DataFrame(X)], axis=1)
    if len(X.shape) > 1:
        df.columns = ['features', 'iqr', 'median_diff', 'ttest', 'score', 'class']
        if ttest:
            df = df.sort_values(by=["ttest"], ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    else:
        df.columns = ['features', 'score']
        df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
    if map_genes:
        df = df.merge(X_map, left_on='features', right_on='Probe Set ID')
        df = df.drop(["features"], axis=1)
    return df


def comparative_score(top_features_pred, top_features_true):
    if len(top_features_pred) != len(top_features_true):
        temp = "The number of samples must be same for both lists."
        raise Exception(temp)
    score = jaccard_score(y_true=top_features_true, y_pred=top_features_pred)
    return score


def outliers_analysis(X, y, regulated_features):
    # Get feature size
    num_features = X.shape[1]

    regulated_features = np.where(regulated_features != 0)[0]

    # Detect outliers
    outliers_dict = dict()
    for group_idx in np.unique(y):
        examples_idx = np.where(y == group_idx)[0]
        q1 = np.percentile(X[examples_idx], q=25, axis=0)
        q3 = np.percentile(X[examples_idx], q=75, axis=0)
        iqr = q3 - q1  # Inter-quartile range
        fence_high = q3 + (1.5 * iqr)
        fence_low = q1 - (1.5 * iqr)
        temp = list()
        for feature_idx in range(num_features):
            temp1 = np.where(X[examples_idx, feature_idx] > fence_high[feature_idx])[0]
            temp2 = np.where(X[examples_idx, feature_idx] < fence_low[feature_idx])[0]
            temp.append(temp1.tolist() + temp2.tolist())
        outliers_dict.update({group_idx: temp})
    del temp, temp1, temp2

    # Calculate the outliers number and properties
    for group_idx, group_item in outliers_dict.items():
        num_outliers = 0
        num_regulated_outliers = 0
        for feature_idx, sample_list in enumerate(group_item):
            if len(sample_list) > 0:
                num_outliers += len(sample_list)
                if feature_idx in regulated_features:
                    num_regulated_outliers += len(sample_list)
                    print(">> Feature: {0}; Group: {1}; Outliers: {2}".format(feature_idx, group_idx, sample_list))
        print("\t>> Average number of outliers per feature: {0:.4f}".format(num_outliers / num_features))
        print("\t>> Average number of outliers per expressed feature: {0:.4f}".format(
            num_regulated_outliers / len(regulated_features)))


def train(num_jobs: int = 4):
    # Arguments
    map_genes = False
    top_k_features = 100
    perform_simulation = True
    analyze_outliers = True
    plot_results = True

    # Load chip data
    hu6800 = pd.read_csv(os.path.join(DATASET_PATH, "HU6800.chip"), sep='\t')
    regulated_features = list()
    # Load expression data
    if perform_simulation:
        print("## Perform simulation studies...")
        X = pd.read_csv(os.path.join(DATASET_PATH, "madsim_syn.csv"), sep=',')
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
        # Load up/down regulated features
        regulated_features = pd.read_csv(os.path.join(DATASET_PATH, "madsim_features.csv"), sep=',')
        regulated_features = regulated_features.to_numpy().squeeze()
        regulated_features[regulated_features < 0] = 1
        if analyze_outliers:
            outliers_analysis(X=X.drop(["class"], axis=1).to_numpy(), y=y, regulated_features=regulated_features)
    else:
        print("## Perform real experimental data analysis...")
        X = pd.read_csv(os.path.join(DATASET_PATH, "leukemia_golub_two.csv"), sep=',')
        y = X["class"].to_numpy()
        features_name = X.drop(["class"], axis=1).columns.to_list()
        X = X.drop(["class"], axis=1).to_numpy()
    count = 1

    print("\t{0})- The cancer outlier profile analysis (COPA)...".format(count))
    df_copa = COPA(q=0.75).fit_predict(X=X, y=y, test_class=1)
    df_copa = sort_features(X=df_copa, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- The outlier-sum statistic (OSS)...".format(count))
    df_os = OutlierSumStatistic(q=0.75, iqr_range=(25, 75), two_sided_test=False).fit_predict(X=X, y=y, test_class=1)
    df_os = sort_features(X=df_os, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- The outlier robust t-statistic (ORT)...".format(count))
    df_ort = OutlierRobustStatistic(q=0.75, iqr_range=(25, 75)).fit_predict(X=X, y=y, normal_class=0, test_class=1)
    df_ort = sort_features(X=df_ort, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- The maximum ordered subset t-statistics (MOST)...".format(count))
    df_most = MOST().fit_predict(X=X, y=y, normal_class=0, test_class=1)
    df_most = sort_features(X=df_most, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- The least sum of ordered subset square t-statistic (LSOSS)...".format(count))
    df_lsoss = LSOSS().fit_predict(X=X, y=y, normal_class=0, test_class=1)
    df_lsoss = sort_features(X=df_lsoss, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- The detection of imbalanced differential signal (DIDS)...".format(count))
    df_dids = DIDS().fit_predict(X=X, y=y, normal_class=0, test_class=1)
    df_dids = sort_features(X=df_dids, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    print("\t{0})- Unraveling cellular heterogeneity by analyzing intra-cellular variation (U-Het)...".format(count))
    df_uhet = UHeT(normalize="robust", q=0.75, iqr_range=(25, 75)).fit_predict(X=X, y=y)
    df_uhet = sort_features(X=df_uhet, features_name=features_name, X_map=hu6800, map_genes=map_genes)
    count += 1

    # df_class = classifiers_results(X=X, y=y, features_name=features_name, num_features=num_features, standardize=True,
    #                                num_jobs=num_jobs, save_path=DATASET_PATH)
    # count += 1

    methods_df = [("COPA", df_copa), ("OS", df_os), ("ORT", df_ort), ("MOST", df_most), ("UHet", df_uhet)]

    if perform_simulation:
        print("\t{0})- Scoring results using true known regulated features...".format(count))
        count += 1
        selected_regulated_features = top_k_features
        temp = regulated_features.sum()
        if selected_regulated_features > temp:
            selected_regulated_features = temp
        print("\t\t>> Number of true up/down regulated features: {0}".format(selected_regulated_features))
        for stat_name, df in methods_df:
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:selected_regulated_features].tolist()]
            top_features_pred = np.zeros((len(regulated_features)))
            top_features_pred[temp] = 1
            score = comparative_score(
                top_features_pred=top_features_pred, top_features_true=regulated_features)
            print("\t\t    --> Jaccard score for {0}: {1:.2f}%".format(stat_name, score))

    if plot_results:
        print("\t{0})- Plot results using top k features...".format(count))
        # plot top k features
        for stat_name, df in methods_df:
            temp = [idx for idx, feature in enumerate(features_name)
                    if feature in df['features'][:top_k_features].tolist()]
            plot_umap(X=X[:, temp], y=y, num_features=top_k_features, standardize=True, num_jobs=num_jobs,
                      suptitle=stat_name.upper(), file_name=stat_name.lower(), save_path=DATASET_PATH)
            plot_boxplot(X=X[:, temp], y=y, features_name=features_name[:top_k_features], num_features=top_k_features,
                         standardize=True)
            plot_clusters(X=X[:, temp], y=y, features_name=features_name[:top_k_features], num_features=top_k_features,
                          standardize=True, cluster_type="spectral", num_clusters=0, num_neighbors=15, min_dist=0,
                          heatmap=True, proportion=True, show_umap=True, num_jobs=num_jobs, suptitle=stat_name.upper(),
                          file_name=stat_name.lower(), save_path=DATASET_PATH)


if __name__ == "__main__":
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
    train(num_jobs=4)
