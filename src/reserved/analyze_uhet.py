"""
Analyze Hetero-Net Results

Author: Caleb Hallinan

License: 
"""

##################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
import umap
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.feature_selection import RFE
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


##################################################################################################

# Function to look at UMAP of results

def UMAP_results(data, y, heteronet_results, n_features, min_dist=0, n_neighbors=15, random_state=0, standardize=True):
    """
    UMAP results from Hetero-Net

    Parameters
    ----------
    data : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    heteronet_results : pd.DataFrame of results from Hetero-Net

    n_features : int
        Number of features to use for UMAP

    min_dist : int (default=0)
        Minimum distance set for UMAP

    n_neighbors : int (default=15)
        Number of nearest neighbors for UMAP

    random_state : int (default=0)
        Random State

    standardize : bool (default=True)
        Standardize data before UMAP

    Returns
    -------
    matplotlib plot of UMAP-ed data with top n_features from Hetero-Net
    """

    # standardize if needed
    if standardize:
        data = zscore(data)

    # check if dataframe, used to subset features later
    # if not isinstance(data, pd.DataFrame):
    #     raise Exception("Input data as pd.Dataframe with features as columns.")

    # make umap and umap data
    reducer_data = umap.UMAP(min_dist=min_dist, random_state=random_state, n_neighbors=n_neighbors)
    umap_data = reducer_data.fit_transform(data[heteronet_results['features'][:n_features]])

    # make column names
    df_umap = pd.DataFrame(umap_data)
    df_umap.columns = ['umap1', 'umap2']

    # add class to dataframe
    df_umap['class'] = y

    # plot figure
    plt.figure(figsize=(6, 5))
    sns.scatterplot(df_umap['umap1'], df_umap['umap2'], hue=df_umap['class'], palette='tab10')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('Using Top ' + str(n_features) + ' Features from Hetero-Net', fontsize=18)
    plt.legend(title="Class")
    sns.despine()


##################################################################################################


# function to look at clustering results

def cluster_results(data, y, heteronet_results, n_features, cluster_type='kmeans', n_clusters=None, min_dist=0,
                    n_neighbors=15, random_state=0, standardize=True, heatmap=False, proportion=False, plot=True):
    """
    Cluster results from Hetero-Net

    Parameters
    ----------
    data : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    heteronet_results : pd.DataFrame of results from Hetero-Net

    n_features : int
        Number of features to use for UMAP

    cluster_type : str (default=kmeans)
        Type of clustering algorithm to use for clustering data

    n_clusters : int (default=None)
        Pre-set number of clusters used, or leave as None to perform Silhoutte Analysis

    min_dist : int (default=0)
        Minimum distance set for UMAP

    n_neighbors : int (default=15)
        Number of nearest neighbors for UMAP

    random_state : int (default=0)
        Random State

    standardize : bool (default=True)
        Standardize data before UMAP

    heatmap : bool (default=False)
        If true, will print heatmaps of clustered data with feature expression per cluster
    Returns
    -------
    matplotlib plot of UMAP-ed data with top n_features from Hetero-Net clustered
    """

    # standardize if needed
    if standardize:
        data = zscore(data)

    # check if dataframe, used to subset features later
    # if not isinstance(data, pd.DataFrame):
    #     raise Exception("Input data as pd.Dataframe with features as columns.")

    # make umap and umap data
    reducer_data = umap.UMAP(min_dist=min_dist, random_state=random_state, n_neighbors=n_neighbors)
    umap_data = reducer_data.fit_transform(data[heteronet_results['features'][:n_features]])

    # make column names
    df_umap = pd.DataFrame(umap_data)
    df_umap.columns = ['umap1', 'umap2']

    # Perform Silhoette Analysis
    silhouette_avg_n_clusters = []

    if n_clusters == None:
        for i in range(2, 10):
            # do clustering first
            if cluster_type == 'kmeans':
                clusterer = KMeans(n_clusters=int(i), random_state=random_state).fit(umap_data)
            if cluster_type == 'sc':
                clusterer = SpectralClustering(n_clusters=int(i), random_state=random_state).fit(umap_data)
            if cluster_type == 'agg':
                clusterer = AgglomerativeClustering(n_clusters=int(i)).fit(umap_data)

            # find silhoette score
            silhouette_avg = silhouette_score(umap_data, clusterer.labels_)

            print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)

            # append and use highest later
            silhouette_avg_n_clusters.append(silhouette_avg)

        # use highest silhoette score for clusters
        tmp = max(silhouette_avg_n_clusters)
        n_clusters = silhouette_avg_n_clusters.index(tmp) + 2  # add 2 here because start clustering at 2
        print('Using ' + str(n_clusters) + ' to cluster data..')
        print("Silhoette Score " + str(tmp))

        if cluster_type == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(umap_data)
            df_umap['cluster'] = kmeans.labels_ + 1
        if cluster_type == 'sc':
            kmeans = SpectralClustering(n_clusters=n_clusters, random_state=random_state).fit(umap_data)
            df_umap['cluster'] = kmeans.labels_ + 1
        if cluster_type == 'agg':
            clusterer = AgglomerativeClustering(n_clusters=int(i)).fit(umap_data)
            df_umap['cluster'] = kmeans.labels_ + 1

    # or do own cluster number
    else:
        if cluster_type == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(umap_data)
            df_umap['cluster'] = kmeans.labels_ + 1
        if cluster_type == 'sc':
            kmeans = SpectralClustering(n_clusters=n_clusters, random_state=random_state).fit(umap_data)
            df_umap['cluster'] = kmeans.labels_ + 1
        tmp = 0

    # add class to dataframe
    df_umap['class'] = y

    # print adjusted Rand score
    # print(df_umap['class'])
    print("Rand Score " + str(rand_score(df_umap['cluster'], df_umap['class'])))
    print("Adjusted Rand Score " + str(adjusted_rand_score(df_umap['cluster'], df_umap['class'])))
    ars = adjusted_rand_score(df_umap['cluster'], df_umap['class'])

    if plot:
        # plot
        plt.figure(figsize=(6, 5))
        sns.scatterplot(df_umap['umap1'], df_umap['umap2'], hue=df_umap['cluster'], palette='Dark2')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('Using Top ' + str(n_features) + ' Features from Hetero-Net', fontsize=18)
        plt.legend(title="Cluster")
        sns.despine()

    # print heatmap if true
    if heatmap:
        ## intensity plot for deephetero
        scaler = MinMaxScaler()
        data = data[heteronet_results['features'][:n_features]]
        scaler.fit(data)
        data = pd.DataFrame(scaler.transform(data))
        data.columns = heteronet_results['features'][:n_features]
        data['cluster'] = df_umap['cluster']

        data_gb = data.groupby('cluster').mean()

        plt.figure(figsize=(2.5, 3))
        # sns.heatmap(to_hm_colon.T,cbar_kws={'label': 'Normalized Average Expression'})
        cg = sns.clustermap(data_gb.T, col_cluster=False, cbar_pos=(.95, .08, .03, .7))
        ax = cg.ax_heatmap
        cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)
        cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)
        ax.set_xlabel('Cluster', fontsize=16)
        ax.set_ylabel('Top Features', fontsize=16)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        cg.ax_cbar.tick_params(labelsize=16)
        cg.ax_cbar.set_ylabel('Normalized Average Expression', fontsize=16)

    # plot proportion bar chart
    if proportion and heatmap:

        data['class'] = y

        dic = {}
        clus_labels = []
        for i in range(1, len(np.unique(data['cluster'])) + 1):
            tmp_df = data[data['cluster'] == i]
            issue = False
            issue_wk = []
            issue_index = []
            for cla in np.unique(data['class']):
                if cla not in np.unique(tmp_df['class']):
                    tmp_df = pd.concat([tmp_df, pd.DataFrame({'class': cla}, index=[0])], ignore_index=True).fillna(0)
                    issue = True
                    issue_wk.append(cla)
            if issue == False:
                dic[i] = list(tmp_df['class'].value_counts().sort_index(ascending=False))
            else:
                tmp_ls = list(tmp_df['class'].value_counts().sort_index(ascending=False))
                for is_wk in issue_wk:
                    tmp_lss = list(np.unique(data['class']))
                    tmp_lss.reverse()
                    issue_index.append(tmp_lss.index(is_wk))
                tmp_arr = np.array(tmp_ls)
                for issue in issue_index:
                    tmp_arr[issue] = 0
                dic[i] = list(tmp_arr)

            clus_labels.append("C" + str(i))

        # proportion of weeks
        df_prop = pd.DataFrame(dic)[::-1]
        df_prop.columns = clus_labels
        df_prop.index = np.unique(data['class'])

        df_prop = df_prop.T.div(df_prop.sum(axis=0), axis=0)

        # plot
        plt.figure(figsize=(6, 5))
        df_prop.plot(kind="bar", stacked=True)
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()

    return tmp, ars, n_clusters


##################################################################################################


# Function to look at boxplots of top features

def boxplot_results(data, y, heteronet_results, n_features, standardize=False, swarmplot=False):
    """
    Boxplots of top features from Hetero-Net

    Parameters
    ----------
    data : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    heteronet_results : pd.DataFrame of results from Hetero-Net

    n_features : int
        Number of features to use for UMAP

    standardize : bool (default=True)
        Standardize data before UMAP

    swarmplot : bool (default=False)
        Determine if swarmplot is also used, helps see distribution better (computationally expensive with large dataset)

    Returns
    -------
    matplotlib boxplots of top features
    """
    # standardize if needed
    if standardize:
        data = zscore(data)

    # add class to dataframe
    data['class'] = y

    # cycle through which features
    for i in range(0, n_features):
        feat_int = heteronet_results['features'][i]
        # plot figure
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=data.loc[:, [feat_int, 'class']], x=data.loc[:, [feat_int, 'class']]['class'], y=feat_int,
                    palette='tab10')
        if swarmplot:
            sns.swarmplot(data=data.loc[:, [feat_int, 'class']], x=data.loc[:, [feat_int, 'class']]['class'],
                          y=feat_int, color='black')
        plt.xlabel('Class')
        sns.despine()


##################################################################################################


# SVM function to compare
def SVM_RFE(data, y, n_features, standardize=True):
    """
    Boxplots of top features from Hetero-Net

    Parameters
    ----------
    data : pd.DataFrame of shape (n_samples, n_features)
        The input samples

    y : array-like of shape (n_samples,)
        The target values

    n_features : int
        Number of features to use for UMAP

    standardize : bool (default=True)
        Standardize data before UMAP

    Returns
    -------
    SVM-RFE results for comparison to Hetero-Net
    """
    # get feature names
    feature_names = data.columns
    # standardize if needed
    if standardize:
        data = zscore(data)

    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(data, y)

    # reindex based on ranking
    feature_names = pd.Series(feature_names)
    feature_names.index = selector.ranking_

    rfe_results = pd.DataFrame(list(feature_names.sort_index().reset_index(drop=True)))
    rfe_results.columns = ['features']

    return rfe_results

##################################################################################################
