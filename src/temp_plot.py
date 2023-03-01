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

result_path = os.path.join(RESULT_PATH, "scRNA")
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
methods = [list(np.repeat(m, total_features.shape[1])) for _, m in methods_name.items()]
methods = np.reshape(methods, (total_features.shape[0] * total_features.shape[1]))
total_features = total_features.reshape((total_features.shape[0] * total_features.shape[1]))
df_features = pd.DataFrame([methods, total_features])
df_features = df_features.T
df_features.columns = ["Methods", "Features"]

# Plot the number of features
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=df_features, width=0.85, palette=palette)
# ax = sns.violinplot(y='Features', x='Methods', data=df_features, palette=palette)
ax.set_xlabel("")
ax.set_ylabel("Number of significant features  \n  found by each method", fontsize=36)
ax.set_xticklabels([])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "10", "100", "1000", "10000", "100000"])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle("6 single cell RNA-seq datasets", fontsize=36)
sns.despine()
plt.tight_layout()

# Plot F1 scores
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Scores', x='Methods', data=df, width=0.85, palette=palette)
# ax = sns.violinplot(y='Scores', x='Methods', data=df_features, scale="count", palette=palette)
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
# ax = sns.violinplot(y='ARI', x='Methods', data=df_features, scale="count", palette=palette)
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
######################## Plasschaert #########################
##############################################################

######################## Human
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                              "plasschaert_human_groups.csv"),
                 sep=',', index_col=0, header=None)
df = df.T
temp = {"secretory>ciliated": "Secretory>Ciliated", "basal": "Basal", "ciliated": "Ciliated",
        "slc16a7+": "SLC16A7+", "basal>secretory": "Basal>Secretory", "secretory": "Secretory",
        "brush+pnec": "Brush+PNEC", "ionocytes": "Ionocytes", "foxn4+": "FOXN4+"}
temp = [temp[item] for item in df["subtypes"].tolist()]
df["subtypes"] = temp

# Use static colors
palette = {"Secretory>Ciliated": "#1f77b4", "Basal": "#ff7f0e", "Ciliated": "#2ca02c",
           "SLC16A7+": "#d62728", "Basal>Secretory": "#9467bd", "Secretory": "#8c564b",
           "Brush+PNEC": "#e377c2", "Ionocytes": "#7f7f7f", "FOXN4+": "#bcbd22"}
distribution = pd.crosstab(df["donors"], df["subtypes"], normalize='index')

plt.figure(figsize=(6, 8))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='donors', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.8,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel("Percentage of all cells", fontsize=32)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=28)
ax.set_xticklabels(["Donor 1", "Donor 2", "Donor 3"], rotation=45, fontsize=32)
ax.legend_.remove()
sns.despine()
plt.tight_layout()

plt.figure(figsize=(4, 2))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='donors', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.9,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel(None)
# manually generate legend
ax.legend(title=None, fontsize=30, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
sns.despine()
plt.tight_layout()

######################## Mouse
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                              "plasschaert_mouse_groups.csv"),
                 sep=',', index_col=0, header=None)
df = df.T
temp = {"secretory": "Secretory", "basal": "Basal", "ciliated": "Ciliated",
        "brush": "Brush", "pnec": "PNEC", "krt4/13+": "KRT4/13+",
        "cycling basal (homeostasis)": "Cycling basal (homeostasis)",
        "ionocytes": "Ionocytes",
        "cycling basal (regeneration)": "Cycling basal (regeneration)",
        "pre-ciliated": "Pre-ciliated"}
temp = [temp[item] for item in df["subtypes"].tolist()]
df["subtypes"] = temp

# Use static colors
palette = {"Secretory": "#1f77b4", "Basal": "#ff7f0e", "Ciliated": "#2ca02c",
           "Brush": "#d62728", "PNEC": "#9467bd", "KRT4/13+": "#8c564b",
           "Cycling basal (homeostasis)": "#e377c2", "Ionocytes": "#7f7f7f",
           "Cycling basal (regeneration)": "#bcbd22", "Pre-ciliated": "#17becf"}
distribution = pd.crosstab(df["timepoints"], df["subtypes"], normalize='index')

plt.figure(figsize=(10, 8))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='timepoints', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.8,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel("Percentage of all cells", fontsize=32)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=28)
ticks = [t.get_text().capitalize() for t in ax.get_xticklabels()]
ax.set_xticklabels(ticks, rotation=45, fontsize=32)
ax.legend_.remove()
sns.despine()
plt.tight_layout()

plt.figure(figsize=(4, 2))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='timepoints', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.9,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel(None)
# manually generate legend
ax.legend(title=None, fontsize=30, ncol=2, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
sns.despine()
plt.tight_layout()

##############################################################
########################## Pulseseq ##########################
##############################################################
top_down_features = 30
features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                    "pulseseq_tuft_vs_ionocyte_phet_b_features.csv"),
                       sep=',', header=None)
features = np.squeeze(features.values.tolist())
enriched_idx = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                        "enriched_terms_ionocytes.txt"),
                           sep='\t', header=None)
enriched_idx.columns = ["Features", "Scores"]
enriched_idx = enriched_idx["Features"].to_list()
temp = []
while len(temp) < top_down_features:
    f = enriched_idx.pop(0)
    if f.startswith("Rp"):
        continue
    temp.append(f)
# while len(temp) < top_down_features:
#     f = enriched_idx.pop(-1)
#     if f.startswith("Rp"):
#         continue
#     temp.append(f)
enriched_idx = [idx for idx, f in enumerate(features) if f in temp]
features = features[enriched_idx]

# load positive true ionocytes and novel one
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pos_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "neg_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
selected_samples = np.append(pos_samples, neg_samples)

# load expression
df = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                              "pulseseq_tuft_vs_ionocyte_phet_b_expression.csv"),
                 sep=',', header=None)
df = df.iloc[selected_samples, enriched_idx]
df.columns = features

plt.figure(figsize=(10, 14))
cg = sns.clustermap(df.T, col_cluster=True, cbar_pos=(.95, .08, .03, .7),
                    cmap="Greys")
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
cg.ax_cbar.tick_params(labelsize=30)
cg.ax_cbar.set_ylabel('Expressions', fontsize=30)
cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=22)
cg.ax_heatmap.set_xticklabels("")
ax = cg.ax_heatmap
# ax.set_xlabel('Samples', fontsize=30)
# ax.set_ylabel('Features', fontsize=30)
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()

df = df.T
temp = {"secretory": "Secretory", "basal": "Basal", "ciliated": "Ciliated",
        "brush": "Brush", "pnec": "PNEC", "krt4/13+": "KRT4/13+",
        "cycling basal (homeostasis)": "Cycling basal (homeostasis)",
        "ionocytes": "Ionocytes",
        "cycling basal (regeneration)": "Cycling basal (regeneration)",
        "pre-ciliated": "Pre-ciliated"}
temp = [temp[item] for item in df["subtypes"].tolist()]
df["subtypes"] = temp
