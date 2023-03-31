import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from utility.file_path import RESULT_PATH

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

##############################################################
######################## Simulated ###########################
##############################################################
# F1 scores
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated",
                              "simulated_normal_methods_outliers_scores.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("F1 scores of each method", fontsize=24)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("F1 scores of each method", fontsize=24)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)

# Features
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated",
                              "simulated_normal_methods_features.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()
df = np.log10(df)

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)

# # Legend
ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10))
ax.legend(title="Methods", title_fontsize=30, fontsize=26, ncol=5,
          loc="lower right", bbox_to_anchor=(1.0, 1.0),
          facecolor="None")

##############################################################
######################### Benchmarks #########################
##############################################################
result_path = os.path.join(RESULT_PATH, "microarray")
suptitle = "6 single cell RNA-seq datasets"
suptitle = "11 microarray gene expression datasets"
methods_name = {"ttest": "t-statistic", "COPA": "COPA", "OS": "OS", "ORT": "ORT",
                "MOST": "MOST", "LSOSS": "LSOSS", "DIDS": "DIDS", "DECO": "DECO",
                "DeltaIQR": "ΔIQR", "PHet_b": "PHet"}

# Use static colors
palette = mcolors.TABLEAU_COLORS
palette = dict([(list(methods_name.values())[idx], item[1]) for idx, item in enumerate(palette.items())
                if idx + 1 <= len(methods_name)])

# Feature scores
files = [f for f in os.listdir(result_path) if f.endswith("_features_scores.csv")]

# DECO
feature_files = sorted([f for f in os.listdir(result_path) if f.endswith("_deco.csv")])
deco_features = []
for f in feature_files:
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    deco_features.append(len(df["features"].to_list()))

# Collect features scores
methods = []
scores = []
for f in files:
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    methods.extend(df.iloc[:, 0].to_list())
    scores.extend(df.iloc[:, 1].to_list())

# Total features
feature_files = [sorted([f for f in os.listdir(result_path) if f.endswith(method.lower() + "_features.csv")])
                 for method, _ in methods_name.items()]
total_features = []
for lst_files in feature_files:
    temp = list()
    for f in lst_files:
        df = pd.read_csv(os.path.join(result_path, f), sep=',', header=None)
        temp.append(len(df.values.tolist()))
    total_features.append(temp)
total_features[7] = deco_features
total_features = np.log10(total_features)

# Dataframe 
df = pd.DataFrame([methods, scores]).T
df.columns = ["Methods", "Scores"]
df["Methods"] = df["Methods"].astype(str)
df["Scores"] = df["Scores"].astype(np.float64)
methods = [list(np.repeat(m, total_features.shape[1])) for _, m in methods_name.items()]
methods = np.reshape(methods, (total_features.shape[0] * total_features.shape[1]))
total_features = total_features.reshape((total_features.shape[0] * total_features.shape[1]))
df_features = pd.DataFrame([methods, total_features])
df_features = df_features.T
df_features.columns = ["Methods", "Features"]

# Plot F1 scores
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Scores', x='Methods', data=df, width=0.85, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("F1 scores of each method", fontsize=36)
ax.set_xticklabels([])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()

# Plot the number of features
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=df_features, width=0.85, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("Number of significant features  \n  found by each method", fontsize=36)
ax.set_xticklabels([])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "10", "100", "1000", "10000", "100000"])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()

# Cluster quality
files = [f for f in os.listdir(result_path) if f.endswith("_cluster_quality.csv")]
methods = list()
scores = list()
for f in files:
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    scores.extend(df.loc[1:]["Scores"].to_numpy())
    methods.extend(df.iloc[1:, 0].to_list())
methods = ["DeltaIQR" if item == "ΔIQR" else item for item in methods]
df_ari = pd.DataFrame([methods, scores]).T
df_ari.columns = ["Methods", "ARI"]
df_ari["Methods"] = df_ari["Methods"].astype(str)
df_ari["ARI"] = df_ari["ARI"].astype(np.float64)
palette["DeltaIQR"] = palette["ΔIQR"]
df_ari.groupby(["Methods"])["ARI"].median()

# Plot ARI
y_values = df_ari["ARI"].values
y_lim = (np.min(y_values), np.max(y_values))
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='ARI', x='Methods', data=df_ari, width=0.85,
                 showfliers=False, palette=palette)
# ax = sns.violinplot(y='ARI', x='Methods', data=df_ari, linewidth = 2,
#                     style ="count", palette=palette)
# ax.set_ylim(y_lim) 
ax.set_xlabel("")
ax.set_ylabel("ARI of each method", fontsize=36)
ax.set_xticklabels([])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()

# import umap
# from sklearn.metrics import pairwise_distances, pairwise_kernels
# num_examples = 1000
# num_choice_examples = 100
# num_features = 10
# X_1 = np.random.normal(0, 0.01, size=(num_examples//4, num_features))
# X_2 = np.random.normal(0.5, 0.2, size=(num_examples//4, num_features))
# X_3 = np.random.normal(-0.5, 0.2, size=(num_examples//4, num_features))
# X_4 = np.random.normal(2, 0.5, size=(num_examples//4, num_features))
# X = np.vstack((X_1, X_2, X_3, X_4))
# X = X / np.linalg.norm(X, axis=1)[:, None]
# temp = num_examples // 4
# pairwise_distances(X, metric="cosine")
# y = [0] * temp + [1] * temp + [2] * temp + [3] * temp
# init = "spectral"
# if num_features >= num_examples:
#     init = "random"
# X_red = umap.UMAP(n_neighbors=30, n_components=2, metric="euclidean",
#                   n_epochs=2000, init=init, min_dist=0.3, spread=100,
#                   n_jobs=10)
# X_red = X_red.fit_transform(X)
# plt.scatter(X_red[:, 0], X_red[:, 1], c=y, cmap='viridis')
