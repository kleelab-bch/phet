import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scanpy as sc
import seaborn as sns

from utility.file_path import RESULT_PATH, DATASET_PATH

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

##############################################################
#################### Plasschaert (Human) #####################
##############################################################
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                              "plasschaert_human_groups.csv"),
                 sep=',', index_col=0, header=None)
df = df.T
temp = {"secretory>ciliated": "Secretory>Ciliated", "basal": "Basal", "ciliated": "Ciliated",
        "slc16a7+": "SLC16A7+", "basal>secretory": "Basal>Secretory", "secretory": "Secretory",
        "brush+pnec": "Brush+PNEC", "ionocytes": "Ionocyte", "foxn4+": "FOXN4+"}
temp = [temp[item] for item in df["subtypes"].tolist()]
df["subtypes"] = temp

# Use static colors
palette = {"Secretory>Ciliated": "#1f77b4", "Basal": "#ff7f0e", "Ciliated": "#2ca02c",
           "SLC16A7+": "#d62728", "Basal>Secretory": "#9467bd", "Secretory": "#8c564b",
           "Brush+PNEC": "#e377c2", "Ionocyte": "#7f7f7f", "FOXN4+": "#bcbd22"}
distribution = pd.crosstab(df["donors"], df["subtypes"], normalize='index')

plt.figure(figsize=(6, 8))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1).stack().reset_index(name='Dist'),
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

# manually generate legend
plt.figure(figsize=(4, 2))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='donors', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.9,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.legend(title=None, fontsize=30, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
sns.despine()
plt.tight_layout()

#### load donors features data
top_down_features = 25
features = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                                    "plasschaert_human_phet_b_features.csv"),
                       sep=',', header=None)
features = np.squeeze(features.values.tolist())
enriched_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                                        "enriched_terms_donors.txt"),
                           sep='\t', header=None)
enriched_idx.columns = ["Features", "Scores"]
enriched_idx = enriched_idx["Features"].to_list()

temp = ["KRT4", "KRT13", "CYP2F1"]
while len(temp) < top_down_features:
    f = enriched_idx.pop(0)
    if f.lower().startswith("rp"):
        continue
    temp.append(f)
while len(temp) < top_down_features * 2:
    f = enriched_idx.pop(-1)
    if f.lower().startswith("rp"):
        continue
    temp.append(f)
enriched_idx = np.unique([idx for idx, f in enumerate(features) if f in temp])
features = features[enriched_idx]

# load donors data
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                                       "donor1.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                                       "donors2and3.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
samples_idx = np.append(pos_samples, neg_samples)
samples_name = ["Donor 1"] * len(pos_samples) + ["Donors 2 and 3"] * len(neg_samples)
samples_name = pd.Series(samples_name)
lut = dict(zip(samples_name.unique(), ["black", "red"]))
row_colors = list(samples_name.map(lut))

# load expression
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_human",
                              "plasschaert_human_phet_b_expression.csv"),
                 sep=',', header=None)
df = df.iloc[samples_idx, enriched_idx]
df.columns = features

# plot heatmap
cg = sns.clustermap(df, figsize=(24, 16), method="average", metric="correlation",
                    z_score=0, row_cluster=False, col_cluster=True, row_colors=row_colors,
                    cmap="rocket_r", cbar_pos=(1, .08, .01, .35))
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)
cg.ax_cbar.tick_params(labelsize=30)
cg.ax_cbar.set_ylabel('Standardized expressions', fontsize=30)
cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(),
                              fontsize=28)
cg.ax_heatmap.set_yticklabels("")
ax = cg.ax_heatmap
ax.yaxis.tick_left()

##############################################################
############################ Patel ###########################
##############################################################
total_features = []
full_features = pd.read_csv(os.path.join(DATASET_PATH, "patel_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f for f in full_features]
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "patel_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
# samples_idx = [idx for idx, _ in enumerate(classes)]
# samples_idx = np.where(classes == 0)[0]
samples_idx = np.where(classes == 1)[0]
samples_name = pd.read_csv(os.path.join(DATASET_PATH,
                                        "patel_types.csv"),
                           sep=',').values.tolist()
samples_name = np.squeeze(samples_name)
samples_name = [item for idx, item in enumerate(samples_name) if idx in samples_idx]
samples_name = pd.Series(samples_name, dtype="category")
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH, "patel_matrix.mtx"))
adata = adata[samples_idx, :]
adata.X = 2 ** adata.X.toarray()
adata.var_names = full_features
samples_name.index = adata.obs.index
adata.obs["clusters"] = samples_name
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell, so that counts become comparable among cells.
sc.pp.normalize_total(adata, target_sum=1e4)
# Logarithmize the data:
sc.pp.log1p(adata)
# IQR
# iqr_list = list()
# for feature_idx, item  in enumerate(full_features):
#     iqr_list.append(iqr(adata.X.toarray()[:, feature_idx]))
# iqr_list = zscore(iqr_list, axis=0)
# iqr_list = pd.Series(iqr_list, dtype=np.float32)
# iqr_list.index = adata.var.index
# Identify highly-variable features.
sc.pp.highly_variable_genes(adata, min_disp=0.5)
# adata.var.dispersions_norm = iqr_list
temp = adata.var["highly_variable"] == True
total_features.extend(list(adata.var_names[temp]))
total_features = sorted(list(set(total_features)))
df = pd.DataFrame(total_features, columns=["features"])
df.to_csv(os.path.join(RESULT_PATH, "patel_hvf_per_conditions.csv"),
          sep=',', index=False)
adata = adata[:, adata.var.highly_variable]
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Plot UMAP 
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'], use_raw=False, add_outline=False,
               legend_loc='on data', legend_fontsize=12, legend_fontoutline=2,
               frameon=False)
