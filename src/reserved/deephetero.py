"""
Author: Caleb Hallinan
Date: 06/21/22
Deep Hetero
Environment: uclid
"""

##################################################################################################


### import packages ###


##################################################################################################


import os

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
import umap
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from keras.models import Model
# from sklearn.manifold import TSNE
# from scipy.spatial import ConvexHull
# from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from scipy.stats import zscore
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.utility.file_path import DATASET_PATH

# import openpyxl
# from sklearn.decomposition import PCA
# from sklearn.decomposition import KernelPCA
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from scipy.stats import mannwhitneyu
# import phate
# from mrmr import mrmr_classif
# import tensorflow_similarity as tfsim

# set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##################################################################################################


### Read in Datasets


##################################################################################################


# colon dataset
colon = pd.read_csv(os.path.join(DATASET_PATH, 'colonTumor.txt'), sep=',', header=None)

colon[0] = colon[0].str.replace('[', '')
colon[2000] = colon[2000].str.replace(']', '').astype(float)

colon.rename(columns={0: 'class'}, inplace=True)
colon['class'] = np.where(colon['class'] == ' negative', 0, 1)
y_colon = list(colon['class'])

## getting rid of genes with high outliers

x = pd.DataFrame(zscore(colon.drop('class', axis=1)))
x.columns = x.columns + 1

x1 = x.loc[:, ~(x > 2.5).any()]
x1 = x1.loc[:, ~(x1 < -2.5).any()]

colon_norm = colon[x1.columns]
colon_norm['class'] = colon['class']

# X_train, X_test, y_train, y_test = train_test_split(colon_norm, y_colon, test_size=0.25, random_state=41)


# abd_p = pd.read_csv('colon_abdurrahman_p.csv', index_col=0)
#
# colon_norm = colon[list(abd_p[abd_p['PValue'] < .1].index)]
# colon_norm['class'] = colon['class']

##################################################################################################

# klein dataset

# klein = pd.read_csv('Klein_expression.txt', sep="\t")
# klein = klein.T
# klein_labels = pd.read_csv('Klein_cell_label.csv')
# new_header = klein.iloc[0]  # grab the first row for the header
# klein = klein[1:]  # take the data less the header row
# klein.columns = new_header  # set the header row as the df header
#
# klein['class'] = list(klein_labels['Cluster'])
#
# pd.Series(klein_labels['Cluster']).value_counts()

##################################################################################################

# zeisel dataset

# zeisel = pd.read_csv('Zeisel_expression.txt', sep="\t")
# zeisel = zeisel.T
# zeisel_labels = pd.read_csv('Zeisel_cell_label.csv')
# new_header = zeisel.iloc[0]  # grab the first row for the header
# zeisel = zeisel[1:]  # take the data less the header row
# zeisel.columns = new_header  # set the header row as the df header
#
# zeisel['class'] = list(zeisel_labels['Label'])
#
# pd.Series(zeisel_labels['Label']).value_counts()

##################################################################################################


#### USER PARAMS ####


##################################################################################################


## Set matrix as dataset of interest

mat = colon_norm
# mat = klein.astype(float)
# mat = zeisel.astype(float)


# Set how many nearest neighbors
num_clusters = 5
# set datatset name for saving
data_name = 'colon'

# creaste save naming scheme
feat_save_name = 'tl_features_' + str(num_clusters) + 'c_' + data_name + '.csv'
feat_emb_save_name = 'tl_embeddings_' + str(num_clusters) + 'c_' + data_name + '.csv'

# how many runs to go do DML
runs = 25

##################################################################################################


### Get desired matrix to input into UMAP/DML ###


##################################################################################################


# Note: Matrix should be (n,m) matrix with features in columns and observation in rows
# Should also include column label of 'class'

# dict to count similarity
dict_ct = {}
dict_cta = {}

# change class to 0, 1
mat['class'] = mat['class'].astype('category').cat.codes
y_mat = mat['class'].astype('category').cat.codes

# obtain feature names
feature_names = mat.drop('class', axis=1).columns

# normalize all data
print('Normalizing Data by Feature..')
mat = pd.DataFrame(zscore(mat.drop('class', axis=1)))
mat.columns = feature_names

# dictionary with keys as feature names
dic = {}
for i in feature_names:
    dic[i] = []

# add class back on to mat
mat['class'] = list(y_mat.astype(int))
print(len(mat['class']))

# print length of class
class_len = len(np.unique(mat['class']))
print("Total number of classes found: " + str(class_len))

##################################################################################################


### BEGIN RUNNING UMAP/DML ###


##################################################################################################


print("Running methodology..")

## Run methodology x amount of times
for n in tqdm(range(runs)):

    # dictionary of matrices by class
    dict_df = {}
    dict_df_full = {}
    for i in range(0, class_len):
        tmp = mat[mat['class'] == i]
        dict_df["class_" + str(i)] = tmp
        dict_df_full["class_" + str(i)] = tmp

    # find length of each dictionary
    length_dict = {key: len(value) for key, value in dict_df.items()}
    # find minimum observations for smallest class
    min_class = min(length_dict.values())

    # print stats at the beginning
    if n == 0:
        print('=' * 80)
        print('Length of each class: ')
        print(length_dict)
        print('Randomly Downsample all classes every run to: ' + str(min_class))
        print('=' * 80)

    # randomly down sample all class
    for key in dict_df:
        dict_df[key] = dict_df[key].sample(n=min_class)
        # drop class, transpose to correct form
        dict_df[key] = dict_df[key].drop(['class'], axis=1).T
        dict_df_full[key] = dict_df_full[key].drop(['class'], axis=1).T
        # sort values
        dict_df[key] = pd.DataFrame(np.sort(dict_df[key]))
        dict_df_full[key] = pd.DataFrame(np.sort(dict_df_full[key]))

    # make final dataframe
    mat_final = pd.concat(dict_df.values(), axis=0)
    mat_final_full = pd.concat(dict_df_full.values(), axis=0)

    ## make umap for clusters

    # set these as default for now
    min_distance = 0
    num_neighbors = 15

    reducer_data = umap.UMAP(min_dist=min_distance, n_neighbors=num_neighbors, random_state=0)
    umap_data = reducer_data.fit_transform(mat_final)
    print('made umap..')

    # make dataframe
    df_umap = pd.DataFrame(umap_data)

    # find which features are which, need index
    feat_class = [str(x) for x in feature_names]
    for num_classes in range(2, class_len + 1):
        tmp_feat_class = [str(x) + '___' + str(num_classes) for x in feature_names]
        feat_class.extend(tmp_feat_class)

    # add feat class to df
    df_umap = pd.concat([df_umap, pd.DataFrame(feat_class)], axis=1)
    df_umap.columns = ['umap1', 'umap2', 'class']

    # kmean for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(umap_data)
    df_umap['cluster'] = kmeans.labels_

    ### User defined here?????
    print('Number of clusters and amount of data in each')
    print(df_umap['cluster'].value_counts())

    ## If we want to plot initial embeddings

    # plot
    # plt.figure(figsize=(6,5))
    # # sns.scatterplot(df_umap['umap1'],df_umap['umap2'], palette = 'tab10')
    # sns.scatterplot(df_umap['umap1'],df_umap['umap2'], hue = df_umap['cluster'], palette = 'tab10')
    # # plt.scatter(df_umap.iloc[closest,:]['umap1'],df_umap.iloc[closest,:]['umap2'], c='black')
    # plt.xlabel('UMAP 1')
    # plt.ylabel('UMAP 2')
    # # plt.title('Lung Raw Data', fontsize=24)
    # plt.legend(title='Cluster')
    # sns.despine()
    # plt.savefig('colon_preDML.svg')
    # df_umap['cluster'].value_counts()

    print('Calculated clusters')

    # find feature length
    feature_len = int(len(mat_final) / class_len)

    # find umap values for each class respectively
    # dict_umap = {}
    # num_low = 0
    # num_high = feature_len
    # for i in range(0,class_len):
    #     dict_umap['class_'+str(i)] = pd.DataFrame(umap_data[num_low:num_high])
    #     num_low += feature_len
    #     num_high += feature_len

    # # make dataframe of umap values for each class
    # umap_final = pd.concat(dict_umap.values(), axis=0)

    # embeddings for DML
    embeddings = mat_final.reset_index(drop=True).astype('float32')
    # cluster labels
    labels = np.array(df_umap['cluster'])

    # creat tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
    dataset = dataset.batch(128)

    # make encoder
    if n == 0:
        n_inputs = mat_final.shape[1]
        # n_inputs = tf.cast(n_inputs,tf.float64)x

        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs / 2)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = n_inputs / 4
        bottleneck = Dense(n_bottleneck)(e)
        # bottleneck = (tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))(bottleneck) # L2 normalize embeddings

        # define autoencoder model
        model = Model(inputs=visible, outputs=bottleneck)
        model.summary()

        # compile autoencoder model
        model.compile(optimizer='adam', loss=tfa.losses.TripletSemiHardLoss())
        # model.compile(optimizer='adam', loss=tfa.losses.TripletHardLoss())
        # model.compile(optimizer='adam', loss=tfa.losses.ContrastiveLoss())
        # model.compile(optimizer='adam', loss=tfsim.losses.MultiSimilarityLoss())
        # model.compile(optimizer='adam', loss=tfa.losses.LiftedStructLoss())

        print("Made model")

    # run encoder
    history = model.fit(dataset, epochs=200, verbose=1)

# predict embeddings and find distances
runs1 = 50
for i in range(0, runs1):

    # dictionary of matrices by class
    dict_df = {}
    dict_df_full = {}
    for i in range(0, class_len):
        tmp = mat[mat['class'] == i]
        dict_df["class_" + str(i)] = tmp
        dict_df_full["class_" + str(i)] = tmp

    # find length of each dictionary
    length_dict = {key: len(value) for key, value in dict_df.items()}
    # find minimum observations for smallest class
    min_class = min(length_dict.values())

    # randomly down sample all class
    for key in dict_df:
        dict_df[key] = dict_df[key].sample(n=min_class)
        # drop class, transpose to correct form
        dict_df[key] = dict_df[key].drop(['class'], axis=1).T
        dict_df_full[key] = dict_df_full[key].drop(['class'], axis=1).T
        # sort values
        dict_df[key] = pd.DataFrame(np.sort(dict_df[key]))
        dict_df_full[key] = pd.DataFrame(np.sort(dict_df_full[key]))

    # make final dataframe
    mat_final = pd.concat(dict_df.values(), axis=0)
    mat_final_full = pd.concat(dict_df_full.values(), axis=0)

    # predict using trained model
    hl_test = pd.DataFrame(model.predict(mat_final))

    # feature length
    feature_len = int(len(mat_final) / class_len)

    # make based on how many classes
    index_needed = list(range(0, feature_len))
    index_final = list(range(0, feature_len))
    # make index for every class
    for i in range(0, class_len - 1):
        index_final.extend(index_needed)

    # make dataframe of umap values for each class
    umap_final = hl_test
    umap_final.index = index_final

    # finding euclidean distance between each class
    eucl_dist_ls = []
    for i in range(0, feature_len):
        tmp_feature = umap_final[umap_final.index == i]
        tmp_ls = []
        for x in range(0, class_len):
            for y in range(0, class_len):
                tmp_ls.append(distance.euclidean(tmp_feature.iloc[x, :], tmp_feature.iloc[y, :]))
                # print(tmp_ls)
        max_eucl_dist = np.sum(tmp_ls)
        eucl_dist_ls.append(max_eucl_dist)

    # add rank number to dic
    for i in range(0, len(eucl_dist_ls)):
        dic[feature_names[i]].append(eucl_dist_ls[i])
    print(len(dic[feature_names[i]]))

##################################################################################################


### Save dataframe ###


##################################################################################################

# final_df = pd.DataFrame(pd.DataFrame(dic).mean(), columns=['score']).reset_index().sort_values(by=["score"], ascending=False).reset_index(drop=True)
# final_df['sd'] = list(pd.DataFrame(dic).std())
final_df = pd.concat([pd.DataFrame(dic).mean(), pd.DataFrame(dic).std()], axis=1).reset_index()
final_df.columns = ['features', 'score', 'score_sd']
final_df = final_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
final_df

final_df.to_csv(feat_save_name, index=False)

# save embeddings
umap_final.to_csv(feat_emb_save_name, index=False)

##################################################################################################

# Fin
