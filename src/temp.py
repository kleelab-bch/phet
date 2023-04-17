import os

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import iqr
from scipy.stats import zscore, gamma
from statsmodels.stats.weightstats import ztest

from utility.file_path import DATASET_PATH, RESULT_PATH

sns.set_theme()
sns.set_theme(style="white")
np.random.seed(seed=12345)

minimum_samples = 5

# Descriptions of the data
file_name = "patel"
expression_file_name = file_name + "_matrix.mtx"
features_file_name = file_name + "_feature_names.csv"
classes_file_name = file_name + "_classes.csv"
subtypes_file = file_name + "_types.csv"
differential_features_file = file_name + "_diff_features.csv"

# Load subtypes file
subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
num_clusters = len(np.unique(subtypes))

# Load features, expression, and class data
features_name = pd.read_csv(os.path.join(
    DATASET_PATH, features_file_name), sep=',')
features_name = features_name["features"].to_list()
y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
y = y["classes"].to_numpy()
X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
X = X.to_df().to_numpy()
np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# Filter data based on counts (CPM)
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
# X = zscore(X, axis=0)
del temp, feature_sums

H = np.zeros((num_features,))
D = np.zeros((num_features,))
HD = np.zeros((num_features,))
examples_i = np.where(y == 0)[0]
examples_j = np.where(y == 1)[0]
for feature_idx in range(num_features):
    iqr1 = iqr(X[examples_i, feature_idx], rng=(25, 75), scale=1.0)
    iqr2 = iqr(X[examples_j, feature_idx], rng=(25, 75), scale=1.0)
    if iqr1 < 0.05 or iqr2 < 0.05:
        continue
    mean1 = np.mean(X[examples_i, feature_idx])
    mean2 = np.mean(X[examples_j, feature_idx])
    statistic, pvalue = ztest(X[examples_i, feature_idx], X[examples_j, feature_idx])
    H[feature_idx] = np.absolute(iqr1 - iqr2)
    D[feature_idx] = np.absolute(statistic)
temp = list()
for feature_type in [H, D]:
    shape, loc, scale = gamma.fit(zscore(feature_type))
    selected_features = np.where((1 - gamma.cdf(zscore(feature_type), shape,
                                                loc=loc, scale=scale)) <= 0.01)[0]
    temp.append(selected_features)
H = temp[0]
D = temp[1]
HD = list(set(temp[0]).intersection(temp[1]))
df_type = pd.DataFrame([H, D, HD], dtype=np.int16).T
df_type.columns = ["H", "D", "HD"]
feature_idx = 2609
sns.boxplot(y=X[:, feature_idx], x=y, width=0.85, showfliers=False,
            showmeans=True, meanprops={"marker": "D",
                                       "markerfacecolor": "white",
                                       "markeredgecolor": "black",
                                       "markersize": "15"})
sns.stripplot(y=X[:, feature_idx], x=y, color="black",
              s=8, linewidth=0, alpha=.4)

# Load subtypes file
df_disp = pd.read_csv(os.path.join(RESULT_PATH,
                                   "patel_hvf_dispersions.csv"),
                      sep=',')
df_disp.columns = ["features", "scores"]

df_disp_conditions = pd.read_csv(os.path.join(RESULT_PATH,
                                              "patel_hvf_dispersions_per_conditions.csv"),
                                 sep=',')
df_disp_conditions.columns = ["features", "scores"]

df_iqr = pd.read_csv(os.path.join(RESULT_PATH,
                                  "patel_hvf_iqr.csv"),
                     sep=',')
df_iqr.columns = ["features", "scores"]

df_iqr_conditions = pd.read_csv(os.path.join(RESULT_PATH,
                                             "patel_hvf_iqr_per_conditions.csv"),
                                sep=',')
df_iqr_conditions.columns = ["features", "scores"]

df = pd.DataFrame([df_disp["scores"].to_list(),
                   df_disp_conditions["scores"].to_list(),
                   df_iqr["scores"].to_list(),
                   df_iqr_conditions["scores"].to_list()]).T
df.index = df_disp["features"].to_list()
df.columns = ["dispersions", "dispersions_conditions", "iqr", "iqr_conditions"]
del df_disp, df_disp_conditions, df_iqr_conditions, df_iqr

temp = list()
for column in df.columns:
    shape, loc, scale = gamma.fit(zscore(df[column]))
    selected_features = np.where((1 - gamma.cdf(zscore(df[column]), shape,
                                                loc=loc, scale=scale)) <= 0.05)[0]
    temp.append(df.iloc[selected_features].index.tolist())
df_confusion = np.zeros((4, 4))
for i in range(4):
    for j in range(i, 4):
        df_confusion[i, j] = len(set(temp[i]).intersection(temp[j]))
df_confusion = pd.DataFrame(df_confusion, columns=df.columns,
                            index=df.columns)
sns.boxplot(y=X[:, features_name.index(temp[2][0])], x=y, width=0.85)
