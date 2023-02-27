import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.file_path import RESULT_PATH

sns.set_theme()
sns.set_style(style='white')

######################## Real data ###########################

result_path = os.path.join(RESULT_PATH, "microarray")
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

# Change names scores
# methods = ["ΔIQR" if item == "DeltaIQR" else item for item in methods]
# methods = ["t-statistic" if item == "ttest" else item for item in methods]

# Dataframe 
df = pd.DataFrame([methods, scores]).T
df.columns = ["Methods", "F1 scores"]
methods = [list(np.repeat(m, total_features.shape[1])) for _, m in methods_name.items()]
methods = np.reshape(methods, (total_features.shape[0] * total_features.shape[1]))
total_features = total_features.reshape((total_features.shape[0] * total_features.shape[1]))
df_features = pd.DataFrame([methods, total_features])
df_features = df_features.T
df_features.columns = ["Methods", "Features"]

# Plot the number of features
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=df_features, width=0.85, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("Number of significant features  \n  found by each method (log10 scale)", fontsize=36)
ax.set_xticklabels([])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("6 single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()

# Plot F1 scores
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='F1 scores', x='Methods', data=df, width=0.85, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("F1 scores of each method", fontsize=36)
ax.set_xticklabels([])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("6 single cell RNA-seq datasets", fontsize=36)
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
df_ari = pd.DataFrame([methods, scores]).T
df_ari.columns = ["Methods", "ARI"]
df_ari.groupby(["Methods"])["ARI"].median()

# Plot ARI
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='ARI', x='Methods', data=df_ari, width=0.85, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("ARI of each method", fontsize=36)
ax.set_xticklabels([])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("6 single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()

overall_scores = []
overall_scores.append(df.groupby(["Methods"])["F1 scores"].mean().tolist())
overall_scores.append(df_ari.groupby(["Methods"])["ARI"].mean().tolist())
overall_scores.append(1 / np.array(df_features.groupby(["Methods"])["Features"].mean().tolist()))

# # Legend
# plt.figure(figsize=(20, 10))
# bplot = sns.boxplot(y='F1 scores', x='Methods', data=df, palette=palette, hue='Methods')
# plt.xticks(fontsize=32, rotation=45)
# plt.yticks(fontsize=32)
# plt.xlabel('Methods', fontsize=36)
# plt.ylabel("F1 scores of each method", fontsize=36)
# plt.suptitle("Results using 6 scRNA datasets", fontsize=36)
# bplot.axes.legend(title="Methods", title_fontsize=30, fontsize=26, 
#                   ncol=1, loc="best", bbox_to_anchor=(1.0, 1.0),
#                   facecolor="None")
# sns.despine()
# plt.tight_layout()


######################## Simulated ###########################

df = pd.read_csv(os.path.join(RESULT_PATH, "simulated_normal_methods_outliers_scores.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()
methods_name = df.index.to_list()
methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
methods_name = ["t-statistic" if item == "ttest" else item for item in methods_name]
df.index = methods_name

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
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated_normal_methods_features.csv"),
                 sep=',', index_col=0)
data_name = df.columns.to_list()
methods_name = df.index.to_list()
methods_name = ["ΔIQR" if item == "DeltaIQR" else item for item in methods_name]
methods_name = ["t-statistic" if item == "ttest" else item for item in methods_name]
df.index = methods_name

temp = [1, 0, 0, 0] * int(len(data_name) / 4)
df_minority = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data_name) / 4)
df_mixed = df[[data_name[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85, figsize=(8, 6))
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)

# # Legend
ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10))
ax.legend(title="Methods", title_fontsize=30, fontsize=26, ncol=5,
          loc="lower right", bbox_to_anchor=(1.0, 1.0),
          facecolor="None")
