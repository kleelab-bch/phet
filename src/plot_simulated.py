import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from utility.file_path import RESULT_PATH, DATASET_PATH

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

methods_name = {"ttest_p": "t-statistic", "ttest_g": "t-statistic+Gamma", "wilcoxon_p": "Wilcoxon",
                "wilcoxon_g": "Wilcoxon+Gamma", "limma_p": "LIMMA", "limma_g": "LIMMA+Gamma",
                "hvf_a": "HVF (composite)", "hvf_c": "HVF (by condition)", "deltahvf": "ΔHVF",
                "deltahvfmean": "ΔHVF+ΔMean", "iqr_a": "IQR (composite)", "iqr_c": "IQR (by condition)",
                "deltaiqr": "ΔIQR", "deltaiqrmean": "ΔIQR+ΔMean", "copa": "COPA", "os": "OS",
                "ort": "ORT", "most": "MOST", "lsoss": "LSOSS", "dids": "DIDS", "deco": "DECO",
                "phet_bh": "PHet (ΔHVF)", "phet_br": "PHet"}
# Use static colors
palette = mcolors.get_named_colors_mapping()
palette = [(item[0], item[1]) for idx, item in enumerate(palette.items())
           if idx % 7 == 0]
palette = dict(palette)
palette = dict([(list(methods_name.values())[idx], item[1])
                for idx, item in enumerate(palette.items())
                if idx + 1 <= len(methods_name)])
palette = sns.color_palette("tab20")
palette.append("#fcfc81")
palette.append("#b5563c")
palette.append("#C724B1")
palette = dict([(item[1], mcolors.to_hex(palette[idx]))
                for idx, item in enumerate(methods_name.items())])

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

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                            figsize=(8, 6), color=palette)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("F1 scores of each method", fontsize=24)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                         figsize=(8, 6), color=palette)
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

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                            figsize=(8, 6), color=palette)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                         figsize=(8, 6), color=palette)
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of significant features \n found by each method", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)

# Legend
ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10), color=palette)
ax.legend(title="Methods", title_fontsize=30, fontsize=26, ncol=5,
          loc="lower right", bbox_to_anchor=(1.0, 1.0),
          facecolor="None")

##############################################################
######################### Benchmarks #########################
##############################################################
result_path = os.path.join(RESULT_PATH, "scRNA")
suptitle = "6 single cell RNA-seq datasets"
# suptitle = "11 microarray gene expression datasets"

# Data names
data_names = pd.read_csv(os.path.join(result_path, "data_names.txt"), sep=',')
data_names = data_names.columns.to_list()

# Feature scores
files = [f for f in os.listdir(result_path) if f.endswith("_features_scores.csv")]

# DECO
feature_files = sorted([f for f in os.listdir(result_path) if f.endswith("_deco_features.csv")])
deco_features = list()
for f in feature_files:
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    deco_features.append(len(df["features"].to_list()))

# Collect features scores
methods = list()
scores = list()
ds_names = list()
for idx, f in enumerate(files):
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    methods.extend(df.iloc[:, 0].to_list())
    scores.extend(df.iloc[:, 1].to_list())
    ds_names.extend(len(df.iloc[:, 0].to_list()) * [data_names[idx]])

# Predicted features
feature_files = [sorted([f for f in os.listdir(result_path)
                         if f.endswith(method.lower() + "_features.csv")])
                 for method, _ in methods_name.items()]
pred_features = list()
for lst_files in feature_files:
    temp = list()
    for f in lst_files:
        df = pd.read_csv(os.path.join(result_path, f), sep=',', header=None)
        temp.append(len(df.values.tolist()))
    pred_features.append(temp)
pred_features[7] = deco_features
pred_features = np.log10(pred_features)

# F1 scores 
df = pd.DataFrame([methods, scores, ds_names]).T
df.columns = ["Methods", "Scores", "Data"]
df["Methods"] = df["Methods"].astype(str)
df["Scores"] = df["Scores"].astype(np.float64)

# Total features
ds_names = data_names * len(methods_name)
methods = [list(np.repeat(m, pred_features.shape[1])) for _, m in methods_name.items()]
methods = np.reshape(methods, (pred_features.shape[0] * pred_features.shape[1]))
pred_features = pred_features.reshape((pred_features.shape[0] * pred_features.shape[1]))
selected_features = pd.DataFrame([methods, pred_features, ds_names])
selected_features = selected_features.T
selected_features.columns = ["Methods", "Features", "Data"]

# Plot F1 scores
min_ds = df[df["Methods"] == "PHet"].sort_values('Scores').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('Scores').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Scores', x='Methods', data=df, width=0.85,
                 palette=palette, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Scores', x='Methods', data=df, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=df[df["Data"] == max_ds], x="Methods", y="Scores",
             color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df[df["Data"] == min_ds], x="Methods", y="Scores",
             color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("F1 scores of each method", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()

# Plot the number of features
min_ds = selected_features[selected_features["Methods"] == "PHet"].sort_values('Features').iloc[0].to_list()[-1]
max_ds = selected_features[selected_features["Methods"] == "PHet"].sort_values('Features').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=selected_features, width=0.85,
                 palette=palette, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Features', x='Methods', data=selected_features, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=selected_features[selected_features["Data"] == max_ds], x="Methods",
             y="Features", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=selected_features[selected_features["Data"] == min_ds], x="Methods",
             y="Features", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("Number of predicted features \n of each method",
              fontsize=36)
ax.set_xticklabels(list())
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
ds_names = list()
for idx, f in enumerate(files):
    df = pd.read_csv(os.path.join(result_path, f), sep=',')
    scores.extend(df.loc[1:]["Scores"].to_numpy())
    methods.extend(df.iloc[1:, 0].to_list())
    ds_names.extend(len(df.iloc[1:, 0].to_list()) * [data_names[idx]])
df_ari = pd.DataFrame([methods, scores, ds_names]).T
df_ari.columns = ["Methods", "ARI", "Data"]
df_ari["Methods"] = df_ari["Methods"].astype(str)
df_ari["ARI"] = df_ari["ARI"].astype(np.float64)
df_ari.groupby(["Methods"])["ARI"].mean()

# Plot ARI
min_ds = df_ari[df_ari["Methods"] == "PHet"].sort_values('ARI').iloc[0].to_list()[-1]
max_ds = df_ari[df_ari["Methods"] == "PHet"].sort_values('ARI').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='ARI', x='Methods', data=df_ari, width=0.85,
                 palette=palette, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='ARI', x='Methods', data=df_ari, color="black", s=10, linewidth=0,
              alpha=.7)
sns.lineplot(data=df_ari[df_ari["Data"] == max_ds], x="Methods",
             y="ARI", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df_ari[df_ari["Data"] == min_ds], x="Methods",
             y="ARI", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("ARI scores of each method", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()

# Legend
plt.figure(figsize=(14, 8))
# Create legend handles manually
handles = [mpl.Patch(color=palette[x], label=x) for x in palette.keys()]
# Create legend
plt.legend(handles=handles, title="Methods", title_fontsize=30, fontsize=26, ncol=5,
           loc="lower right", bbox_to_anchor=(1.0, 1.0),
           facecolor="None")

##############################################################
############# ARI vs Sample Size vs Feature Size #############
##############################################################
sns.set_theme(style="ticks")
result_scrna_path = os.path.join(RESULT_PATH, "scRNA")
result_microarray_path = os.path.join(RESULT_PATH, "microarray")
ds_files = ["allgse412", "bc_ccgse3726", "bladdergse89", "braintumor",
            "gastricgse2685", "glioblastoma", "leukemia_golub",
            "lunggse1987", "lung", "mll", "srbct", "baron1", "camp1",
            "darmanis", "li", "patel", "yan"]
# data_names = ["GSE412", "GSE3726", "GSE89", "Braintumor", "GSE2685", 
#               "Glioblastoma", "Leukemia", "GSE1987", "Lung", "MLL", "SRBCT",
#               "Baron", "Camp", "Darmanis", "Li", "Patel", "Yan"]
# Various scores
methods = list()
ari_scores = list()
f1_scores = list()
pred_features = list()
feature_size = list()
sample_size = list()
ds_names = list()

for result_path in [result_microarray_path, result_scrna_path]:
    data_names = pd.read_csv(os.path.join(result_path, "data_names.txt"), sep=',')
    data_names = data_names.columns.to_list()
    files = [f for f in os.listdir(result_path) if f.endswith("_cluster_quality.csv")]
    for idx, f in enumerate(files):
        df = pd.read_csv(os.path.join(result_path, f), sep=',')
        ari_scores.extend(df.loc[1:]["Scores"].to_numpy())
        methods.extend(df.iloc[1:, 0].to_list())
        ds_names.extend(len(df.iloc[1:, 0].to_list()) * [data_names[idx]])
    files = [f for f in os.listdir(result_path) if f.endswith("_features_scores.csv")]
    for idx, f in enumerate(files):
        df = pd.read_csv(os.path.join(result_path, f), sep=',')
        f1_scores.extend(df.loc[0:]["Scores"].to_numpy())
    files = [[f for f in os.listdir(result_path) if f.endswith(method.lower() + "_features.csv")]
             for method, _ in methods_name.items()]
    files = np.array(files)
    for f_idx in range(files.shape[1]):
        for m_idx in range(files.shape[0]):
            f = files[m_idx, f_idx]
            if f.endswith("_deco_features.csv"):
                df = pd.read_csv(os.path.join(result_path, f), sep=',')
                pred_features.append(len(df["features"].to_list()))
            else:
                df = pd.read_csv(os.path.join(result_path, f), sep=',', header=None)
                pred_features.append(len(df.values.tolist()))

for f in ds_files:
    temp = pd.read_csv(os.path.join(DATASET_PATH,
                                    f + "_feature_names.csv"), sep=',')
    feature_size.extend(len(methods_name) * [temp.shape[0]])
    temp = pd.read_csv(os.path.join(DATASET_PATH, f + "_classes.csv"), sep=',')
    sample_size.extend(len(methods_name) * [temp.shape[0]])

pred_features = np.log10(pred_features)
feature_size = np.array(feature_size)
sample_size = np.array(sample_size)
df = pd.DataFrame([methods, ari_scores, f1_scores, pred_features,
                   feature_size, sample_size, ds_names]).T
df.columns = ["Methods", "ARI", "F1", "Predicted features",
              "Feature size", "Sample size", "Data"]
df["Methods"] = df["Methods"].astype(str)
df["ARI"] = df["ARI"].astype(np.float64)
df["F1"] = df["F1"].astype(np.float64)
df["Predicted features"] = df["Predicted features"].astype(np.float64)

# Set up a grid to plot survival probability against several variables
g = sns.PairGrid(df, y_vars="ARI",
                 x_vars=["Methods", "Feature size", "Sample size"],
                 height=5, aspect=.5)
# Draw a seaborn pointplot onto each Axes
g.map(sns.scatterplot)
g.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
for ax in g.axes.flatten():
    ax.tick_params(rotation=90)
g.set(ylim=(0, 1))
g.fig.set_size_inches(16, 4)
sns.despine(fig=g.fig, left=True)

# Plot ARI
min_ds = df[df["Methods"] == "PHet"].sort_values('ARI').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('ARI').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='ARI', x='Methods', data=df, width=0.85,
                 palette=palette, showfliers=False)
sns.swarmplot(y='ARI', x='Methods', data=df, color="black", s=10, linewidth=0,
              alpha=.7)
sns.pointplot(y="ARI", x="Methods", data=df, scale=1.3,
              errwidth=5, markers="D", color="#343d46")
sns.lineplot(data=df[df["Data"] == max_ds], x="Methods",
             y="ARI", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df[df["Data"] == min_ds], x="Methods",
             y="ARI", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("ARI scores of each method", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("17 microarray and single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()

# Plot F1
min_ds = df[df["Methods"] == "PHet"].sort_values('F1').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('F1').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='F1', x='Methods', data=df, width=0.85,
                 palette=palette, showfliers=False)
sns.swarmplot(y='F1', x='Methods', data=df, color="black", s=10, linewidth=0,
              alpha=.7)
sns.pointplot(y="F1", x="Methods", data=df, scale=1.3,
              errwidth=5, markers="D", color="#343d46")
sns.lineplot(data=df[df["Data"] == max_ds], x="Methods",
             y="F1", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df[df["Data"] == min_ds], x="Methods",
             y="F1", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("F1 scores of each method", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("17 microarray and single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()

# Plot Predicted features
min_ds = df[df["Methods"] == "PHet"].sort_values('Predicted features').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('Predicted features').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Predicted features', x='Methods', data=df, width=0.85,
                 palette=palette, showfliers=False)
sns.swarmplot(y='Predicted features', x='Methods', data=df, color="black",
              s=10, linewidth=0, alpha=.7)
sns.pointplot(y="Predicted features", x="Methods", data=df, scale=1.3,
              errwidth=5, markers="D", color="#343d46")
sns.lineplot(data=df[df["Data"] == max_ds], x="Methods",
             y="Predicted features", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df[df["Data"] == min_ds], x="Methods",
             y="Predicted features", color="red", linewidth=3,
             linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("Number of predicted features \n of each method",
              fontsize=36)
ax.set_xticklabels(list())
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "10", "100", "1000", "10000", "100000"])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("17 microarray and single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()


##############################################################
############# TEMP #############
##############################################################

######### Intracluster -- Measuring distance between points in a cluster. Lower values are better intraclustering quality

def complete_diameter_distance(X, y, penalty: float = 0.05, metric: str = "euclidean",
                               num_jobs: int = 2):
    '''
    The farthest distance between two samples in a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            score += penalty
            continue
        D = pairwise_distances(X=X[examples_idx], metric=metric, n_jobs=num_jobs)
        D = D / D.sum(0)
        score += D.max()
    return score


def average_diameter_distance(X, y, penalty: float = 0.05, metric: str = "euclidean",
                              num_jobs: int = 2):
    '''
    The average distance between all samples in a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            score += penalty
            continue
        D = pairwise_distances(X=X[examples_idx], metric=metric, n_jobs=num_jobs)
        D = D / D.sum(0)
        D = np.triu(D).sum()
        score += 1 / (len(examples_idx) * (len(examples_idx) - 1)) * D
    return score


def centroid_diameter_distance(X, y, penalty: float = 0.05, metric: str = "euclidean",
                               num_jobs: int = 2):
    '''
    The double of average distance between samples and the center of a cluster.
    '''
    clusters = np.unique(y)
    score = 0
    for cluster_idx in clusters:
        examples_idx = np.where(y == cluster_idx)[0]
        if len(examples_idx) <= 1:
            score += penalty
            continue
        cluster_centers = X[examples_idx].sum(0) / len(examples_idx)
        cluster_centers = cluster_centers[None, :]
        D = pairwise_distances(X=X[examples_idx], Y=cluster_centers, metric=metric,
                               n_jobs=num_jobs)
        D = D / D.sum()
        score += 2 * (D.sum() / len(examples_idx))
    return score


########## Intercluster -- Measuring distance between clusters. Higher values are better.
def single_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The closest distance between two samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / D.sum(0)
            score += D.min()
    return score


def maximum_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The farthest distance between two samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / D.sum(0)
            score += D.max()
    return score


def average_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The average distance between all samples in clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            D = pairwise_distances(X=X[examples_i], Y=X[examples_j], metric=metric,
                                   n_jobs=num_jobs)
            D = D / D.sum(0)
            score += D.sum() / (len(examples_i) * len(examples_j))
    return score


def centroid_linkage_distance(X, y, metric: str = "euclidean", num_jobs: int = 2):
    '''
    The distance between centers of clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            center_i = X[examples_i].sum(0) / len(examples_i)
            center_i = center_i[None, :]
            center_j = X[examples_j].sum(0) / len(examples_j)
            center_j = center_j[None, :]
            D = pairwise_distances(X=center_i, Y=center_j, metric=metric,
                                   n_jobs=num_jobs)
            score += D.flatten()[0]
    return score


def wards_distance(X, y):
    '''
    The different deviation between a group of 2 considered clusters and a 
    "reputed" cluster joining those 2 clusters.
    '''
    num_clusters = np.unique(y).shape[0]
    score = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            examples_i = np.where(y == i)[0]
            examples_j = np.where(y == j)[0]
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                continue
            center_i = X[examples_i].sum(0) / len(examples_i)
            center_j = X[examples_j].sum(0) / len(examples_j)
            temp = (2 * len(examples_i) * len(examples_j)) / (len(examples_i) + len(examples_j))
            if len(examples_i) <= 1 or len(examples_j) <= 1:
                score += np.sqrt(temp * np.sum((center_i - center_j) ** 2)) * penalty
                continue
            score += np.sqrt(temp * np.sum((center_i - center_j) ** 2))
    return score


result_path = os.path.join(RESULT_PATH, "temp")
minimum_samples = 5
metric = "euclidean"
num_jobs = 2
ds_files = ["allgse412", "bc_ccgse3726", "bladdergse89", "braintumor",
            "gastricgse2685", "glioblastoma", "leukemia_golub",
            "lunggse1987", "lung", "mll", "srbct", "baron1", "camp1",
            "darmanis", "li", "patel", "yan"]
for ds_name in ds_files:
    # Expression, classes, subtypes, donors, timepoints Files
    expression_file_name = ds_name + "_matrix.mtx"
    features_file_name = ds_name + "_feature_names.csv"
    classes_file_name = ds_name + "_classes.csv"
    subtypes_file = ds_name + "_types.csv"
    # Load subtypes file
    subtypes = pd.read_csv(os.path.join(DATASET_PATH, subtypes_file), sep=',').dropna(axis=1)
    subtypes = [str(item[0]).lower() for item in subtypes.values.tolist()]
    num_clusters = len(np.unique(subtypes))
    # Load features, expression, and class data
    features_name = pd.read_csv(os.path.join(DATASET_PATH, features_file_name), sep=',')
    features_name = features_name["features"].to_list()
    y = pd.read_csv(os.path.join(DATASET_PATH, classes_file_name), sep=',')
    y = y["classes"].to_numpy()
    X = sc.read_mtx(os.path.join(DATASET_PATH, expression_file_name))
    X = X.to_df().to_numpy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Filter data
    num_examples, num_features = X.shape
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
    feature_ids = dict([(feature_idx, idx)
                        for idx, feature_idx in enumerate(feature_ids)])
    num_examples, num_features = X.shape
    del temp, feature_sums

    labels_true = np.unique(subtypes)
    labels_true = dict([(item, idx) for idx, item in enumerate(labels_true)])
    labels_true = [labels_true[item] for item in subtypes]
    labels_true = np.array(labels_true)

    list_scores = list()
    for method, _ in methods_name.items():
        temp_feature = ds_name + "_" + method.lower() + "_features.csv"
        temp_cluster = ds_name + "_" + method.lower() + "_clusters.csv"
        labels_pred = pd.read_csv(os.path.join(result_path, temp_cluster), sep=',', header=0)
        labels_pred = labels_pred["Cluster"].to_list()
        labels_pred = np.array(labels_pred)
        if method != "deco":
            selected_features = pd.read_csv(os.path.join(result_path, temp_feature), sep=',', header=None)
            selected_features = selected_features[0].to_list()
        else:
            selected_features = pd.read_csv(os.path.join(result_path, temp_feature), sep=',')
            selected_features = selected_features["features"].to_list()
            selected_features = [features_name[feature_ids[item]] for item in selected_features]

        selected_features = [idx for idx, feature in enumerate(features_name)
                             if feature in selected_features]

        intracluster_complete = complete_diameter_distance(X=X[:, selected_features], y=labels_pred,
                                                           metric=metric, num_jobs=num_jobs)
        intracluster_average = average_diameter_distance(X=X[:, selected_features], y=labels_pred,
                                                         metric=metric, num_jobs=num_jobs)
        intracluster_centroid = centroid_diameter_distance(X=X[:, selected_features], y=labels_pred,
                                                           metric=metric, num_jobs=num_jobs)
        intercluster_single = single_linkage_distance(X=X[:, selected_features], y=labels_pred,
                                                      metric=metric, num_jobs=num_jobs)
        intercluster_maximum = maximum_linkage_distance(X=X[:, selected_features], y=labels_pred,
                                                        metric=metric, num_jobs=num_jobs)
        intercluster_average = average_linkage_distance(X=X[:, selected_features], y=labels_pred,
                                                        metric=metric, num_jobs=num_jobs)
        intercluster_centroid = centroid_linkage_distance(X=X[:, selected_features], y=labels_pred,
                                                          metric=metric, num_jobs=num_jobs)
        intercluster_wards = wards_distance(X=X[:, selected_features], y=labels_pred)
        silhouette = silhouette_score(X=X[:, selected_features], labels=labels_pred, metric=metric)
        ari = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)
        ami = adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)

        list_scores.append([intracluster_complete, intracluster_average, intracluster_centroid,
                            intercluster_single, intercluster_maximum, intercluster_average,
                            intercluster_centroid, intercluster_wards, silhouette, ari, ami])
    df = pd.DataFrame(list_scores, index=list(methods_name.values()),
                      columns=["Complete Diameter Distance", "Average Diameter Distance", 
                               "Centroid Diameter Distance", "Single Linkage Distance", 
                               "Maximum Linkage Distance", "Average Linkage Distance",
                               "Centroid Linkage Distance", "Ward's Distance", "Silhouette", 
                               "Adjusted Rand Index", "Adjusted Mutual Info"])
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, ds_name + "_cluster_quality.csv"),
              sep=",")
