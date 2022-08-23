"""
Author: Caleb Hallinan
Date: 10/29/21
Test new method
Environment: uclid
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from umap_eucl import umap_eucl
# from heteronet import heteronet
# from heteronet import heteronet_sample
from scipy.stats import zscore
from scipy.stats import iqr
import time
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# import importlib
# import analyze_heteronet
# import heteronet
# importlib.reload(analyze_heteronet)
# importlib.reload(heteronet)

# chip
hu6800 = pd.read_csv("/media/bch_drive/Public/CalebHallinan/heteronet/data/hu6800_chip.txt", sep='\t')


################################################################################

# luk = pd.read_csv('/media/bch_drive/Public/CalebHallinan/heteronet/data/golub.csv', sep=',')

# # get rid of some stuff
# luk = luk.drop(['Samples','BM.PB','Gender','Source','tissue.mf'],axis=1)
# # get 3 class labe
# y_luk3 = list(luk['cancer'])

# y_luk = []
# for i in y_luk3:
#     if "all" in i:
#         y_luk.append("all")
#     else:
#         y_luk.append('aml')

# luk.rename(columns={'cancer':'class'}, inplace=True)
# luk['class'] = y_luk

# ## getting rid of genes with high outliers

# luk_norm = luk.copy()
# luk_hetero = luk_norm.copy()
# luk_norm['class3'] = y_luk3

# luk_norm['class3'] = np.where(luk_norm['class3'] == 'allB','ALL-B',luk_norm['class3'])
# luk_norm['class3'] = np.where(luk_norm['class3'] == 'allT','ALL-T',luk_norm['class3'])
# luk_norm['class3'] = np.where(luk_norm['class3'] == 'aml','AML',luk_norm['class3'])
# pd.Series(luk_norm['class3']).value_counts()

# X = luk_hetero.copy()

def heteroiqr(X, standardize='none', random_state=0, ttest=False, map_genes=False):
    """
    Hetero-Net Function

    Perform Deep Metric Learning with UMAP-based clustering to find subpopulations of classes

    Read more in the USER GUIDE

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples

    standardize : bool, default=True
        Standardizes data using zscore (NOTE: test more?)

    Attributes
    ----------
    features_ : list
        The features found by the algorithm. Ranked in order of importance

    score_ : list
        The score found by the algorithm for each feature


    References
    ----------
    NOTE: MINE


    Examples
    ----------
    NOTE: TODO
    """

    # time function
    start_time = time.time()

    # set cat code for y
    # y = y.astype('category').cat.codes
    num_classes = len(np.unique(X['class']))
    n_features = X.shape[1]
    unique_class_labels = np.unique(X['class'])

    print("Number of Features: " + str(n_features))
    print("Number of Classes: " + str(num_classes))

    # standarize zscore
    if standardize == 'zscore':
        # X_resampled = zscore(X_resampled)
        X_final = zscore(X.loc[:, X.columns != 'class'])
        X_final['class'] = X['class']

    # standardize minmaxscaler
    # else:
    #     scaler = MinMaxScaler()
    #     scaler.fit(X)
    #     X = scaler.transform(X)
    else:
        X_final = X.loc[:, X.columns != 'class']
        X_final['class'] = X['class']
        print('no standard')

    # make transposed matrix with shape (feat per class, observation per class)

    # find mean and iqr difference between genes
    var_ls = []
    mean_ls = []
    ttest_ls = []
    what_class = []
    for i in range(0, n_features - 1):
        tmp_ls = []
        tmp_mean_ls = []
        tmp_ttest_ls = []
        for x in range(0, num_classes):
            tmp_class1 = unique_class_labels[x]
            for y in range(x + 1, num_classes):
                tmp_class2 = unique_class_labels[y]
                tmp_ls.append(iqr(X_final[X_final['class'] == tmp_class1].iloc[:, i]) - iqr(
                    X_final[X_final['class'] == tmp_class2].iloc[:, i]))
                tmp_mean_ls.append(np.mean(X_final[X_final['class'] == tmp_class1].iloc[:, i]) - np.mean(
                    X_final[X_final['class'] == tmp_class2].iloc[:, i]))
                tmp_ttest_ls.append(stats.ttest_ind(X_final[X_final['class'] == tmp_class1].iloc[:, i],
                                                    X_final[X_final['class'] == tmp_class2].iloc[:, i])[0])

        # check if negative to seperate classes for later
        if max(tmp_ls) <= 0:
            what_class.append(0)
        else:
            what_class.append(1)

        # append the top variance
        var_ls.append(max(np.abs(tmp_ls)))
        mean_ls.append(max(np.abs(tmp_mean_ls)))
        ttest_ls.append(max(np.abs(tmp_ttest_ls)))

    final_df = pd.concat([pd.DataFrame(var_ls)], axis=1)
    final_df.columns = ['iqr']
    final_df['median_diff'] = mean_ls
    final_df['ttest'] = ttest_ls

    # scale lists to give weights to each
    mean_ls_scaled = np.array((mean_ls - np.min(mean_ls)) / (np.max(mean_ls) - np.min(mean_ls)))
    # mean_ls_scaled = np.array((mean_ls - np.min(ttest_ls))/(np.max(ttest_ls)-np.min(ttest_ls))) ## testing t test
    var_ls_scaled = np.array((var_ls - np.min(var_ls)) / (np.max(var_ls) - np.min(var_ls)))

    final_df['score'] = np.array(mean_ls) + np.array(var_ls)
    final_df['features'] = X.drop('class', axis=1).columns
    final_df['class_diff'] = what_class

    if ttest:
        # final_df = final_df.sort_values(by=["ttest"]).reset_index(drop=True)
        final_df = final_df.sort_values(by=["ttest"], ascending=False).reset_index(drop=True)

    else:
        final_df = final_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    if map_genes:
        # map gene labels
        final_df = final_df.merge(hu6800, left_on='features', right_on='Probe Set ID')

    # print time
    print("--- %s seconds ---" % round(time.time() - start_time, 2))

    return final_df


def copa(X, standardize='median', percentile=75, random_state=0, disease_label=1, map_genes=False):
    # time function
    start_time = time.time()

    # set cat code for y
    # y = y.astype('category').cat.codes
    num_classes = len(np.unique(X['class']))
    n_features = X.shape[1]
    unique_class_labels = np.unique(X['class'])

    print("Number of Features: " + str(n_features))
    print("Number of Classes: " + str(num_classes))

    # standarize zscore
    if standardize == 'zscore':
        X_final = zscore(X.loc[:, X.columns != 'class'])
        X_final['class'] = X['class']

    # standarize median
    if standardize == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X.drop('class', axis=1))
        X_final = pd.DataFrame(scaler.transform(X.drop('class', axis=1)))
        X_final.columns = X.drop('class', axis=1).columns
        X_final['class'] = X['class']

    else:
        X_final = X.loc[:, X.columns != 'class']
        X_final['class'] = X['class']
        print('no standard')

    # do copa
    # qr uses only disease state
    qr = np.percentile(X_final[X_final['class'] == disease_label].drop('class', axis=1), percentile)
    med = X_final.drop('class', axis=1).median()
    mad = abs(X_final.loc[:, X_final.columns != 'class'] - X_final.loc[:,
                                                           X_final.columns != 'class'].median()).median() * 1.4826
    copa = (qr - med) / mad

    final_df = pd.concat([pd.DataFrame(copa)], axis=1).reset_index()
    final_df.columns = ['features', 'score']
    final_df = final_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    if map_genes:
        # map gene labels
        final_df = final_df.merge(hu6800, left_on='features', right_on='Probe Set ID')

    return final_df


def os(X, standardize='median', percentile=.75, random_state=0, disease_label=1, map_genes=False):
    # time function
    start_time = time.time()

    # set cat code for y
    # y = y.astype('category').cat.codes
    num_classes = len(np.unique(X['class']))
    n_features = X.shape[1]
    unique_class_labels = np.unique(X['class'])

    print("Number of Features: " + str(n_features))
    print("Number of Classes: " + str(num_classes))

    # standarize zscore
    if standardize == 'zscore':
        X_final = zscore(X.loc[:, X.columns != 'class'])
        X_final['class'] = X['class']

    # standarize median
    if standardize == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X.drop('class', axis=1))
        X_final = pd.DataFrame(scaler.transform(X.drop('class', axis=1)))
        X_final.columns = X.drop('class', axis=1).columns
        X_final['class'] = X['class']

    # standarize median
    if standardize == 'median':
        med = X.drop('class', axis=1).median()
        mad = abs(X.loc[:, X.columns != 'class'] - X.loc[:, X.columns != 'class'].median()).median() * 1.4826
        X_final = (X.loc[:, X.columns != 'class'] - med) / mad
        X_final['class'] = X['class']

    else:
        X_final = X.loc[:, X.columns != 'class']
        X_final['class'] = X['class']
        print('no standard')

    # do os
    # qr uses only disease state
    qr = X_final.drop('class', axis=1).quantile(percentile)
    iqr = X_final.drop('class', axis=1).quantile(.75) - X_final.drop('class', axis=1).quantile(.25)

    qr_iqr = qr + iqr

    # indicator matrix
    I = X_final[X_final[X_final['class'] == disease_label].drop('class', axis=1) > qr_iqr].drop('class', axis=1)

    # find os stat
    os = I.sum(axis=0)

    final_df = pd.concat([pd.DataFrame(os)], axis=1).reset_index()
    final_df.columns = ['features', 'score']
    final_df = final_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    if map_genes:
        # map gene labels
        final_df = final_df.merge(hu6800, left_on='features', right_on='Probe Set ID')

    return final_df


def ort(X, standardize='median', percentile=.75, random_state=0, nondisease_label=0, disease_label=1, map_genes=False):
    # time function
    start_time = time.time()

    # set cat code for y
    # y = y.astype('category').cat.codes
    num_classes = len(np.unique(X['class']))
    n_features = X.shape[1]
    unique_class_labels = np.unique(X['class'])

    print("Number of Features: " + str(n_features))
    print("Number of Classes: " + str(num_classes))

    # standarize zscore
    if standardize == 'zscore':
        X_final = zscore(X.loc[:, X.columns != 'class'])
        X_final['class'] = X['class']

    # standarize median
    if standardize == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X.drop('class', axis=1))
        X_final = pd.DataFrame(scaler.transform(X.drop('class', axis=1)))
        X_final.columns = X.drop('class', axis=1).columns
        X_final['class'] = X['class']

    # standarize median
    if standardize == 'median':
        med = X.drop('class', axis=1).median()
        mad = abs(X.loc[:, X.columns != 'class'] - X.loc[:, X.columns != 'class'].median()).median() * 1.4826
        X_final = (X.loc[:, X.columns != 'class'] - med) / mad
        X_final['class'] = X['class']

    # do os
    # qr uses only disease state
    qr = X_final[X_final['class'] == nondisease_label].drop('class', axis=1).quantile(percentile)
    iqr = X_final[X_final['class'] == nondisease_label].drop('class', axis=1).quantile(.75) - X_final[
        X_final['class'] == nondisease_label].drop('class', axis=1).quantile(.25)

    qr_iqr = qr + iqr

    I = X_final[X_final[X_final['class'] == disease_label].drop('class', axis=1) > qr_iqr].drop('class', axis=1)

    os = I.sum(axis=0)

    final_df = pd.concat([pd.DataFrame(os)], axis=1).reset_index()
    final_df.columns = ['features', 'score']
    final_df = final_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    if map_genes:
        # map gene labels
        final_df = final_df.merge(hu6800, left_on='features', right_on='Probe Set ID')

    return final_df
