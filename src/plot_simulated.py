import os

import matplotlib.colors as mcolors
import matplotlib.patches as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from PIL import Image

from utility.file_path import RESULT_PATH, DATASET_PATH
from utility.utils import clustering_performance

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

methods_name = {"ttest_p": "t-statistic", "ttest_g": "t-statistic+Gamma", "wilcoxon_p": "Wilcoxon",
                "wilcoxon_g": "Wilcoxon+Gamma", "ks_p": "KS", "ks_g": "KS+Gamma", "limma_p": "LIMMA",
                "limma_g": "LIMMA+Gamma", "dispersion_a": "Dispersion (composite)",
                "dispersion_c": "Dispersion (by condition)", "deltadispersion": "ΔDispersion",
                "deltadispersionmean": "ΔDispersion+ΔMean", "iqr_a": "IQR (composite)",
                "iqr_c": "IQR (by condition)", "deltaiqr": "ΔIQR", "deltaiqrmean": "ΔIQR+ΔMean",
                "copa": "COPA", "os": "OS", "ort": "ORT", "most": "MOST", "lsoss": "LSOSS",
                "dids": "DIDS", "deco": "DECO", "phet_bd": "PHet (ΔDispersion)", "phet_br": "PHet"}
selected_methods = ["t-statistic+Gamma", "Wilcoxon+Gamma", "KS+Gamma", "LIMMA+Gamma", "ΔDispersion",
                    "ΔDispersion+ΔMean", "ΔIQR", "ΔIQR+ΔMean", "COPA", "OS", "ORT", "MOST",
                    "LSOSS", "DIDS", "DECO", "PHet (ΔDispersion)", "PHet"]
# Use static colors
PALETTE = sns.color_palette("tab20")
PALETTE.append("#fcfc81")
PALETTE.append("#C724B1")
PALETTE.append("#fcfc81")
PALETTE.append("#b5563c")
PALETTE.append("#C724B1")
PALETTE.append("#606c38")
PALETTE.append("#283618")
PALETTE = dict([(item[1], mcolors.to_hex(PALETTE[idx]))
                for idx, item in enumerate(methods_name.items())])

####################################################################################
###                      Simulated Evaluations and Plots                         ###
####################################################################################
# F1 scores
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated",
                              "simulated_normal_methods_outliers_scores.csv"),
                 sep=',', index_col=0)
data = df.columns.to_list()

temp = [1, 0, 0, 0] * int(len(data) / 4)
df_minority = df[[data[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data) / 4)
df_mixed = df[[data[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                            figsize=(10, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("F1 score ", fontsize=24)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_f1.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                         figsize=(10, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("F1 score", fontsize=24)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_mixed_case_control_f1.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_minority.loc[selected_methods].T.plot.bar(rot=0, legend=False,
                                                  align='center', width=0.85,
                                                  figsize=(8, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("F1 score ", fontsize=24)
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_f1_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_mixed.loc[selected_methods].T.plot.bar(rot=0, legend=False,
                                               align='center', width=0.85,
                                               figsize=(8, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("F1 score", fontsize=24)
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_mixed_case_control_f1_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

# Features
df = pd.read_csv(os.path.join(RESULT_PATH, "simulated",
                              "simulated_normal_methods_features.csv"),
                 sep=',', index_col=0)
data = df.columns.to_list()
df = np.log10(df)

temp = [1, 0, 0, 0] * int(len(data) / 4)
df_minority = df[[data[idx] for idx, item in enumerate(temp) if item == 1]]
temp = [0, 1, 0, 0] * int(len(data) / 4)
df_mixed = df[[data[idx] for idx, item in enumerate(temp) if item == 1]]

ax = df_minority.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                            figsize=(10, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("Number of predicted features", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_features.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_mixed.T.plot.bar(rot=0, legend=False, align='center', width=0.85,
                         figsize=(10, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of predicted features", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_control_features.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_minority.loc[selected_methods].T.plot.bar(rot=0, legend=False,
                                                  align='center', width=0.85,
                                                  figsize=(8, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case samples)", fontsize=24)
ax.set_ylabel("Number of predicted features", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["1/20", "3/20", "5/20", "7/20", "9/20"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_features_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

ax = df_mixed.loc[selected_methods].T.plot.bar(rot=0, legend=False,
                                               align='center', width=0.85,
                                               figsize=(8, 6), color=PALETTE)
ax.set_xlabel("Number of outliers (case and control samples)", fontsize=22)
ax.set_ylabel("Number of predicted features", fontsize=24)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xticklabels(["2/40", "6/40", "10/40", "14/40", "18/40"])
ax.tick_params(axis='both', labelsize=24)
file_name = "simulated_minority_case_control_features_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.tight_layout()
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

# Legend
ax = df_mixed.T.plot.bar(rot=0, figsize=(20, 10), color=PALETTE)
ax.legend(title="Methods", title_fontsize=30, fontsize=26, ncol=5,
          loc="lower right", bbox_to_anchor=(1.0, 1.0),
          facecolor="None")

####################################################################################
###               Combine multiple images into a single mega image               ###
####################################################################################
folder_name = "scRNA"
result_path = os.path.join(RESULT_PATH, folder_name)
if folder_name == "microarray":
    list_data = ["allgse412", "bc_ccgse3726", "bladdergse89", "braintumor",
                 "gastricgse2685", "glioblastoma", "leukemia_golub", "lunggse1987",
                 "lung", "mll", "srbct"]
else:
    list_data = ["baron1", "camp1", "darmanis", "li", "patel", "yan"]

for data in list_data:
    # need to manually do list to get correct order
    files = [os.path.join(result_path, data + '_all_subtypes_umap.png'),
             os.path.join(result_path, data + '_ttest_p_subtypes_umap.png'),
             os.path.join(result_path, data + '_ttest_g_subtypes_umap.png'),
             os.path.join(result_path, data + '_wilcoxon_p_subtypes_umap.png'),
             os.path.join(result_path, data + '_wilcoxon_g_subtypes_umap.png'),
             os.path.join(result_path, data + '_ks_p_subtypes_umap.png'),
             os.path.join(result_path, data + '_ks_g_subtypes_umap.png'),
             os.path.join(result_path, data + '_limma_p_subtypes_umap.png'),
             os.path.join(result_path, data + '_limma_g_subtypes_umap.png'),
             os.path.join(result_path, data + '_dispersion_a_subtypes_umap.png'),
             os.path.join(result_path, data + '_dispersion_c_subtypes_umap.png'),
             os.path.join(result_path, data + '_deltadispersion_subtypes_umap.png'),
             os.path.join(result_path, data + '_deltadispersionmean_subtypes_umap.png'),
             os.path.join(result_path, data + '_iqr_a_subtypes_umap.png'),
             os.path.join(result_path, data + '_iqr_c_subtypes_umap.png'),
             os.path.join(result_path, data + '_deltaiqr_subtypes_umap.png'),
             os.path.join(result_path, data + '_deltaiqrmean_subtypes_umap.png'),
             os.path.join(result_path, data + '_copa_subtypes_umap.png'),
             os.path.join(result_path, data + '_os_subtypes_umap.png'),
             os.path.join(result_path, data + '_ort_subtypes_umap.png'),
             os.path.join(result_path, data + '_most_subtypes_umap.png'),
             os.path.join(result_path, data + '_lsoss_subtypes_umap.png'),
             os.path.join(result_path, data + '_dids_subtypes_umap.png'),
             os.path.join(result_path, data + '_deco_subtypes_umap.png'),
             os.path.join(result_path, data + '_phet_bd_subtypes_umap.png'),
             os.path.join(result_path, data + '_phet_br_subtypes_umap.png')]

    # add ct variables to put each image in the correct spot
    ct = 0
    ct1 = 0
    # for loop going through files
    for idx in range(len(files)):
        temp_img = Image.open(files[idx])
        # create new image
        if idx == 0:
            # get temp image size, all should be the same
            width, height = temp_img.size
            new_image = Image.new('RGB', (width * 4, height * 7))
            new_image.paste(temp_img, (0, 0))
        # puts the last two images in the middle of the rows and not the beginning
        elif idx == len(files) - 2:
            new_image.paste(temp_img, (int(width * 1), height * 6))
        elif idx == len(files) - 1:
            new_image.paste(temp_img, (int(width * 2), height * 6))
        # if % 4 = 0 then we want to go to new row
        elif idx % 4 == 0:
            if idx > 2:
                ct += 1
            ct1 = 0
            new_image.paste(temp_img, (width * ct1, height * ct))
        # if % 4 = 1 then we want to stay row and next column
        elif idx % 4 == 1:
            ct1 += 1
            new_image.paste(temp_img, (width * ct1, height * ct))
        # if % 4 = 2 then we want to stay row and next column
        elif idx % 4 == 2:
            ct1 += 1
            new_image.paste(temp_img, (width * ct1, height * ct))
        # if else then it will go in last column
        else:
            ct1 += 1
            new_image.paste(temp_img, (width * ct1, height * ct))

    white_background = Image.new(mode="RGBA", size=(width, height), color="white")
    new_image.paste(white_background, (int(width * 0), height * 6))
    new_image.paste(white_background, (int(width * 3), height * 6))
    new_image.save(os.path.join(RESULT_PATH, data + '.png'))

####################################################################################
###             Microarray and scRNA Benchmark Evaluations and Plots             ###
####################################################################################
folder_name = "microarray"
result_path = os.path.join(RESULT_PATH, folder_name)
if folder_name == "microarray":
    suptitle = "11 microarray gene expression datasets"
else:
    suptitle = "6 single cell RNA-seq datasets"
# Data names
data_names = pd.read_csv(os.path.join(result_path, "data_names.txt"), sep=',')
data_names = data_names.columns.to_list()

# Feature scores
files = [f for f in os.listdir(result_path) if f.endswith("_features_scores.csv")]

# DECO
feature_files = sorted([f for f in os.listdir(result_path) if f.endswith("_deco_features.csv")])
deco_features = list()
for idx in feature_files:
    df = pd.read_csv(os.path.join(result_path, idx), sep=',', header=None)
    deco_features.append(len(df[0].values.tolist()))

# Collect features scores
methods = list()
scores = list()
ds_names = list()
for idx, file in enumerate(files):
    df = pd.read_csv(os.path.join(result_path, file), sep=',')
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
    for idx in lst_files:
        df = pd.read_csv(os.path.join(result_path, idx), sep=',', header=None)
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
df_features = pd.DataFrame([methods, pred_features, ds_names])
df_features = df_features.T
df_features.columns = ["Methods", "Features", "Data"]

# Plot F1 scores
min_ds = df[df["Methods"] == "PHet"].sort_values('Scores').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('Scores').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Scores', x='Methods', data=df, width=0.85,
                 palette=PALETTE, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Scores', x='Methods', data=df, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=df[df["Data"] == max_ds], x="Methods", y="Scores",
             color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df[df["Data"] == min_ds], x="Methods", y="Scores",
             color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("F1 score", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()
file_name = folder_name.lower() + "_f1.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

temp = list()
for m in selected_methods:
    temp.extend(np.where(df["Methods"] == m)[0])
selected_df = df.iloc[temp]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Scores', x='Methods', data=selected_df, width=0.85,
                 palette=PALETTE, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Scores', x='Methods', data=selected_df, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=selected_df[selected_df["Data"] == max_ds], x="Methods", y="Scores",
             color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=selected_df[selected_df["Data"] == min_ds], x="Methods", y="Scores",
             color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("F1 score", fontsize=36)
ax.set_xticklabels(list())
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()
file_name = folder_name.lower() + "_f1_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

# Plot the number of features
min_ds = df_features[df_features["Methods"] == "PHet"].sort_values('Features').iloc[0].to_list()[-1]
max_ds = df_features[df_features["Methods"] == "PHet"].sort_values('Features').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=df_features, width=0.85,
                 palette=PALETTE, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Features', x='Methods', data=df_features, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=df_features[df_features["Data"] == max_ds], x="Methods",
             y="Features", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=df_features[df_features["Data"] == min_ds], x="Methods",
             y="Features", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("Number of predicted features",
              fontsize=36)
ax.set_xticklabels(list())
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "10", "100", "1000", "10000", "100000"])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()
file_name = folder_name.lower() + "_features.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

temp = list()
for m in selected_methods:
    temp.extend(np.where(df_features["Methods"] == m)[0])
selected_df = df_features.iloc[temp]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='Features', x='Methods', data=selected_df, width=0.85,
                 palette=PALETTE, showfliers=False, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": "15"})
sns.swarmplot(y='Features', x='Methods', data=selected_df, color="black", s=10,
              linewidth=0, alpha=.7)
sns.lineplot(data=selected_df[selected_df["Data"] == max_ds], x="Methods",
             y="Features", color="green", linewidth=3, linestyle='dashed')
sns.lineplot(data=selected_df[selected_df["Data"] == min_ds], x="Methods",
             y="Features", color="red", linewidth=3, linestyle='dotted')
ax.set_xlabel("")
ax.set_ylabel("Number of predicted features",
              fontsize=36)
ax.set_xticklabels(list())
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "10", "100", "1000", "10000", "100000"])
ax.tick_params(axis='both', labelsize=30)
plt.suptitle(suptitle, fontsize=36)
sns.despine()
plt.tight_layout()
file_name = folder_name.lower() + "_features_front.png"
file_path = os.path.join(RESULT_PATH, file_name)
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

# Cluster quality
files = [f for f in os.listdir(result_path) if f.endswith("_cluster_quality.csv")]
metrics = ['Complete Diameter Distance', 'Average Diameter Distance', 'Centroid Diameter Distance',
           'Single Linkage Distance', 'Maximum Linkage Distance', 'Average Linkage Distance',
           'Centroid Linkage Distance', 'Ward\'s Distance', 'Silhouette', 'Homogeneity',
           'Completeness', 'V-measure', 'Adjusted Rand Index', 'Adjusted Mutual Info']
metrics_name = ['complete_diameter_distance', 'average_diameter_distance', 'centroid_diameter_distance',
                'single_linkage_distance', 'maximum_linkage_distance', 'average_linkage_distance',
                'centroid_linkage_distance', 'wards_distance', 'silhouette', 'homogeneity',
                'completeness', 'v_measure', 'adjusted_rand_index', 'adjusted_mutual_info']
for column_idx, column in enumerate(metrics):
    methods = list()
    scores = list()
    ds_names = list()
    for idx, file in enumerate(files):
        df = pd.read_csv(os.path.join(result_path, file), sep=',', index_col=0)
        scores.extend(df[column].to_numpy()[1:])
        methods.extend(df.index.to_list()[1:])
        ds_names.extend(len(df.index.to_list()[1:]) * [data_names[idx]])
    df_cluster = pd.DataFrame([methods, scores, ds_names]).T
    df_cluster.columns = ["Methods", column, "Data"]
    df_cluster["Methods"] = df_cluster["Methods"].astype(str)
    min_ds = df_cluster[df_cluster["Methods"] == "PHet"].sort_values(column).iloc[0].to_list()[-1]
    max_ds = df_cluster[df_cluster["Methods"] == "PHet"].sort_values(column).iloc[-1].to_list()[-1]
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(y=column, x='Methods', data=df_cluster, width=0.85,
                     palette=PALETTE, showfliers=False, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white",
                                "markeredgecolor": "black", "markersize": "15"})
    sns.swarmplot(y=column, x='Methods', data=df_cluster, color="black", s=10, linewidth=0,
                  alpha=.7)
    sns.lineplot(data=df_cluster[df_cluster["Data"] == max_ds], x="Methods",
                 y=column, color="green", linewidth=3, linestyle='dashed')
    sns.lineplot(data=df_cluster[df_cluster["Data"] == min_ds], x="Methods",
                 y=column, color="red", linewidth=3, linestyle='dotted')
    ax.set_xlabel("")
    ax.set_ylabel(column.capitalize(), fontsize=36)
    ax.set_xticklabels(list())
    ax.tick_params(axis='both', labelsize=30)
    plt.suptitle(suptitle, fontsize=36)
    sns.despine()
    plt.tight_layout()
    file_name = folder_name.lower() + "_" + str(metrics_name[column_idx]).lower() + ".png"
    file_path = os.path.join(RESULT_PATH, file_name)
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")

    if column == "Adjusted Rand Index":
        temp = list()
        for m in selected_methods:
            temp.extend(np.where(df_cluster["Methods"] == m)[0])
        df_cluster = df_cluster.iloc[temp]
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(y=column, x='Methods', data=df_cluster, width=0.85,
                         palette=PALETTE, showfliers=False, showmeans=True,
                         meanprops={"marker": "D", "markerfacecolor": "white",
                                    "markeredgecolor": "black", "markersize": "15"})
        sns.swarmplot(y=column, x='Methods', data=df_cluster, color="black", s=10, linewidth=0,
                      alpha=.7)
        sns.lineplot(data=df_cluster[df_cluster["Data"] == max_ds], x="Methods",
                     y=column, color="green", linewidth=3, linestyle='dashed')
        sns.lineplot(data=df_cluster[df_cluster["Data"] == min_ds], x="Methods",
                     y=column, color="red", linewidth=3, linestyle='dotted')
        ax.set_xlabel("")
        ax.set_ylabel(column.capitalize(), fontsize=36)
        ax.set_xticklabels(list())
        ax.tick_params(axis='both', labelsize=30)
        plt.suptitle(suptitle, fontsize=36)
        sns.despine()
        plt.tight_layout()
        file_name = folder_name.lower() + "_" + str(metrics_name[column_idx]).lower() + "_front.png"
        file_path = os.path.join(RESULT_PATH, file_name)
        plt.savefig(file_path)
        plt.clf()
        plt.cla()
        plt.close(fig="all")

# Legend
plt.figure(figsize=(16, 10))
handles = [mpl.Patch(color=PALETTE[x], label=x) for x in PALETTE.keys()]
plt.legend(handles=handles, title="Methods", title_fontsize=30, fontsize=26, ncol=3,
           loc="lower right", bbox_to_anchor=(1.0, 1.0),
           facecolor="None", frameon=False)
sns.despine()
plt.tight_layout()
file_path = os.path.join(RESULT_PATH, "legends.png")
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

# Legend
plt.figure(figsize=(22, 8))
handles = [mpl.Patch(color=PALETTE[x], label=x) for x in PALETTE.keys()]
plt.legend(handles=handles, title="Methods", title_fontsize=30, fontsize=26, ncol=5,
           loc="lower right", bbox_to_anchor=(1.0, 1.0),
           facecolor="None", frameon=False)
sns.despine()
plt.tight_layout()
file_path = os.path.join(RESULT_PATH, "legends_full.png")
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

plt.figure(figsize=(22, 6))
handles = [mpl.Patch(color=PALETTE[x], label=x) for x in PALETTE.keys()
           if x in selected_methods]
plt.legend(handles=handles, title="Methods", title_fontsize=30, fontsize=26, ncol=6,
           loc="lower right", bbox_to_anchor=(1.0, 1.0),
           facecolor="None", frameon=False)
sns.despine()
plt.tight_layout()
file_path = os.path.join(RESULT_PATH, "legends_front.png")
plt.savefig(file_path)
plt.clf()
plt.cla()
plt.close(fig="all")

####################################################################################
###                Total Avergae Scores of Each Method and Plots                 ###
####################################################################################

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
    for idx, idx in enumerate(files):
        df = pd.read_csv(os.path.join(result_path, idx), sep=',')
        ari_scores.extend(df.loc[1:]["Scores"].to_numpy())
        methods.extend(df.iloc[1:, 0].to_list())
        ds_names.extend(len(df.iloc[1:, 0].to_list()) * [data_names[idx]])
    files = [f for f in os.listdir(result_path) if f.endswith("_features_scores.csv")]
    for idx, idx in enumerate(files):
        df = pd.read_csv(os.path.join(result_path, idx), sep=',')
        f1_scores.extend(df.loc[0:]["Scores"].to_numpy())
    files = [[f for f in os.listdir(result_path) if f.endswith(method.lower() + "_features.csv")]
             for method, _ in methods_name.items()]
    files = np.array(files)
    for f_idx in range(files.shape[1]):
        for m_idx in range(files.shape[0]):
            idx = files[m_idx, f_idx]
            if idx.endswith("_deco_features.csv"):
                df = pd.read_csv(os.path.join(result_path, idx), sep=',')
                pred_features.append(len(df["features"].to_list()))
            else:
                df = pd.read_csv(os.path.join(result_path, idx), sep=',', header=None)
                pred_features.append(len(df.values.tolist()))

for idx in ds_files:
    temp = pd.read_csv(os.path.join(DATASET_PATH,
                                    idx + "_feature_names.csv"), sep=',')
    feature_size.extend(len(methods_name) * [temp.shape[0]])
    temp = pd.read_csv(os.path.join(DATASET_PATH, idx + "_classes.csv"), sep=',')
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

# Plot ARI
min_ds = df[df["Methods"] == "PHet"].sort_values('ARI').iloc[0].to_list()[-1]
max_ds = df[df["Methods"] == "PHet"].sort_values('ARI').iloc[-1].to_list()[-1]
plt.figure(figsize=(14, 8))
ax = sns.boxplot(y='ARI', x='Methods', data=df, width=0.85,
                 palette=PALETTE, showfliers=False)
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
                 palette=PALETTE, showfliers=False)
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
                 palette=PALETTE, showfliers=False)
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
            df_features = pd.read_csv(os.path.join(result_path, temp_feature), sep=',', header=None)
            df_features = df_features[0].to_list()
        else:
            df_features = pd.read_csv(os.path.join(result_path, temp_feature), sep=',')
            df_features = df_features["features"].to_list()
            df_features = [features_name[feature_ids[item]] for item in df_features]

        df_features = [idx for idx, feature in enumerate(features_name)
                       if feature in df_features]
        list_scores = clustering_performance(X=X, labels_true=labels_true, labels_pred=labels_pred,
                                             num_jobs=2)
    df = pd.DataFrame(list_scores, index=list(methods_name.values()),
                      columns=["Complete Diameter Distance", "Average Diameter Distance",
                               "Centroid Diameter Distance", "Single Linkage Distance",
                               "Maximum Linkage Distance", "Average Linkage Distance",
                               "Centroid Linkage Distance", "Ward's Distance", "Silhouette",
                               "Homogeneity", "Completeness", "V-measure", "Adjusted Rand Index",
                               "Adjusted Mutual Info"])
    df.to_csv(path_or_buf=os.path.join(RESULT_PATH, ds_name + "_cluster_quality.csv"),
              sep=",")
