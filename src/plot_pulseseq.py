import copy
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
from scipy.stats import permutation_test, gamma, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

from utility.file_path import RESULT_PATH, DATASET_PATH
from utility.utils import clustering_performance_wo_ground_truth

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

##############################################################
#### Plot Heatmap based on Top Expressed PHet's Features #####
##############################################################
ionocytes_features = pd.read_csv(os.path.join(DATASET_PATH,
                                              "cell_type_category_human_rna_ionocytes.tsv"),
                                 sep='\t')
ionocytes_features = ionocytes_features["Gene"].to_list()
ionocytes_features = [item.lower() for item in ionocytes_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_tuft_vs_ionocyte_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
temp = [f for f in phet_features if f.lower() in ionocytes_features]
temp.extend(["Cftr", "Foxi1", "Pou2f3", "Egr1"])

enriched_idx = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                        "enriched_terms_pulseseq_ionocytes.txt"),
                           sep='\t', header=None)
enriched_idx.columns = ["Features", "Scores"]
enriched_idx = enriched_idx["Features"].to_list()
top_down_features = 25
to_add = (2 * top_down_features - len(temp)) // 2 + len(temp)
while len(temp) < to_add:
    f = enriched_idx.pop(0)
    if f.lower().startswith("rp"):
        continue
    temp.append(f)
while len(temp) < top_down_features * 2:
    f = enriched_idx.pop(-1)
    if f.lower().startswith("rp"):
        continue
    temp.append(f)
enriched_idx = [idx for idx, f in enumerate(phet_features) if f in temp or f.lower() in ionocytes_features]
phet_features = phet_features[enriched_idx]
# load positive true ionocytes and novel one
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_pos_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_neg_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
samples_idx = np.append(pos_samples, neg_samples)
group_name = ["Ionocytes"] * len(pos_samples) + ["Unknown"] * len(neg_samples)
group_name = pd.Series(group_name)
lut = dict(zip(group_name.unique(), ["black", "red"]))
row_colors = list(group_name.map(lut))

# load expression
df = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                              "pulseseq_tuft_vs_ionocyte_phet_b_expression.csv"),
                 sep=',', header=None)
df = df.iloc[samples_idx, enriched_idx]
df.columns = [f.upper() for f in phet_features]

# plot heatmap
cg = sns.clustermap(df, figsize=(24, 20), method="average", metric="correlation",
                    z_score=0, row_cluster=False, col_cluster=True,
                    row_colors=row_colors, cmap="RdBu_r", cbar_pos=(1, .08, .01, .35))
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
####### Diff. Tests of Ionocyte & Unconventional Cells #######
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_tuft_vs_ionocyte_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# Load samples indices
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_pos_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_neg_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
samples_idx = np.append(pos_samples, neg_samples)
group_name = ["Ionocyte"] * len(pos_samples) + ["Unconventional"] * len(neg_samples)
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in phet_features]]
adata.var_names = [f.upper() for idx, f in enumerate(full_features) if f in phet_features]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
n_components = 50
pca = PCA(n_components=n_components)
pca.fit(X=adata.to_df())
temp = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(temp <= 0.9)[0][-1] + 1
pca = PCA(n_components=n_components)
adata.obsm["X_pca"] = pca.fit_transform(X=adata.to_df())
sc.pp.neighbors(adata, n_neighbors=25, n_pcs=n_components)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.4, key_added="clusters")
adata.X = X
# Rename clusters
new_cluster_names = ['Ionocyte', 'Unconventional']
adata.rename_categories('clusters', new_cluster_names)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon')
# Store the ranked genes into positive ionocytes and negative unknown 
# for GOEA using GOATOOLS
pos = [];
neg = []
for item in adata.uns["rank_genes_groups"]["names"]:
    pos.append(item[0])
    neg.append(item[1])
pos = pd.DataFrame(pos)
neg = pd.DataFrame(neg)
pos.to_csv(os.path.join(RESULT_PATH, "pulseseq", "pulseseq_pos_ionocytes_features.csv"), sep=',',
           index=False, header=False)
neg.to_csv(os.path.join(RESULT_PATH, "pulseseq", "pulseseq_neg_ionocytes_features.csv"), sep=',',
           index=False, header=False)


# Permutaion test
# because our statistic is vectorized, we pass `vectorized=True`
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


pvalues = []
for feature in phet_features:
    item = adata.to_df()[feature.upper()]
    x = item[pos_samples.astype(str)]
    y = item[neg_samples.astype(str)]
    res = permutation_test((x, y), statistic, vectorized=True,
                           n_resamples=10000, alternative='two-sided')
    pvalues.append(res.pvalue)
rejected, pvals_corrected, _, _ = multipletests(pvals=pvalues, alpha=0.01, method="fdr_bh")
temp = np.nan_to_num(pvals_corrected, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
significant_features = []
for idx, condition in enumerate(rejected):
    if condition:
        significant_features.append((phet_features[idx], pvals_corrected[idx]))
significant_features = pd.DataFrame(significant_features, columns=["Feature", "Adj_pvalue_FDR"])
significant_features.sort_values("Adj_pvalue_FDR", inplace=True)
significant_features.to_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_ionocytes_features_permutaion.csv"),
                            sep=',', index=False, header=False)
shape, loc, scale = gamma.fit(temp)
significant_features = np.where((1 - gamma.cdf(zscore(temp), shape, loc=loc, scale=scale)) <= 0.01)[0]
significant_features = np.array(phet_features)[significant_features].tolist()
significant_features = [f.upper() for f in significant_features]
significant_features = pd.DataFrame(significant_features)
significant_features.to_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_ionocytes_features_permutaion_gamma.csv"),
                            sep=',', index=False, header=False)

##############################################################
######## Analysis of Ionocytes & Unconventional Cells ########
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_tuft_vs_ionocyte_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# Load markers
markers_dict = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_markers.csv"), sep=',').replace(np.nan, -1)
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item == -1:
            continue
        if item.upper() in phet_features:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
ionocytes_features = pd.read_csv(os.path.join(DATASET_PATH,
                                              "cell_type_category_human_rna_ionocytes.tsv"),
                                 sep='\t')
ionocytes_features = ionocytes_features["Gene"].to_list()
ionocytes_features = [item.upper() for item in ionocytes_features]
ionocytes_features.extend(markers_dict['Ionocyte'])
ionocytes_features = [f for f in phet_features if f in ionocytes_features]
ionocytes_features = list(sorted(set(ionocytes_features)))
# Load samples indices
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_pos_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_neg_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
samples_idx = np.append(pos_samples, neg_samples)
group_name = ["Ionocyte"] * len(pos_samples) + ["Unconventional"] * len(neg_samples)
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in phet_features]]
adata.var_names = [f.upper() for idx, f in enumerate(full_features) if f in phet_features]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
X = copy.deepcopy(adata.X)
# Computing the neighborhood graph
n_components = 50
pca = PCA(n_components=n_components)
pca.fit(X=adata.to_df())
temp = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(temp <= 0.9)[0][-1] + 1
pca = PCA(n_components=n_components)
adata.obsm["X_pca"] = pca.fit_transform(X=adata.to_df())
sc.pp.neighbors(adata, n_neighbors=25, n_pcs=n_components)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
adata.X = X
# Filter markers
selected_markers_dict = {'Basal': ['KRT5', "KRT8"],
                         'Club': ["NFIA", "MUC5B", "SCGB1A1", "NOTCH2"],
                         'Tuft': ["RGS13", "ALOX5AP", "PTPRC", "POU2F3"],
                         'PNEC': ['ASCL1', 'ASCL2'],
                         'Ionocyte': ['CFTR', 'FOXI1', 'ASCL3'],
                         'Ciliated': ["FOXJ1", "CETN2", "TUBA1A", 'CDHR3'],
                         'Goblet': ['FOXQ1']}
temp_dict = {}
for key, items in selected_markers_dict.items():
    temp = []
    for item in items:
        if item in adata.var_names:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
selected_markers_dict = temp_dict
ionocytes_features.extend(["CFTR", "FOXI1", "POU2F3", "EGR1"])
ionocytes_features = np.unique(ionocytes_features)
ionocytes_features = [item for item in ionocytes_features if item in adata.var.index]
# Plot UMAP
with plt.rc_context({'figure.figsize': (8, 5), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'],
               use_raw=False, add_outline=False, legend_loc='on data',
               frameon=False, title='Clusters', legend_fontsize=24,
               palette={"Ionocyte": "black", "Unconventional": "red"})
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['CFTR', 'FOXI1', 'ASCL3', 'EGR1'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2, frameon=False)
# Plot differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon')
with plt.rc_context({'figure.figsize': (8, 5), 'figure.labelsize': '20',
                     'axes.titlesize': '24', 'axes.labelsize': '20',
                     'xtick.labelsize': '14', 'ytick.labelsize': '14'}):
    sc.pl.rank_genes_groups(adata, n_genes=20, fontsize=18, sharey=False,
                            **{"axes.xlabel": "Ranking"})
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['CFTR', 'FOXI1', 'ASCL3', 'EGR1'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, ionocytes_features, groupby='clusters', title=None,
                  colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
rng = np.random.default_rng()
temp = rng.permuted(np.arange(adata[adata.obs["clusters"] == "Ionocyte"].shape[0]))
df = adata[adata.obs["clusters"] == "Ionocyte"].to_df().sample(adata[adata.obs["clusters"] == "Ionocyte"].shape[0])

adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, ionocytes_features, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, swap_axes=False, figsize=(11, 8))
    sc.pl.heatmap(adata, markers_dict["Ionocyte"], groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, var_group_rotation=45, swap_axes=False,
                  figsize=(11, 6))
    sc.pl.heatmap(adata, markers_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, var_group_rotation=45, swap_axes=False,
                  figsize=(11, 3))

##############################################################
## Disp. based Analysis of Ionocytes & Unconventional Cells ##
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Load markers
markers_dict = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_markers.csv"), sep=',').replace(np.nan, -1)
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item == -1:
            continue
        if item.upper() in full_features:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
ionocytes_features = pd.read_csv(os.path.join(DATASET_PATH,
                                              "cell_type_category_human_rna_ionocytes.tsv"),
                                 sep='\t')
ionocytes_features = ionocytes_features["Gene"].to_list()
ionocytes_features = [item.upper() for item in ionocytes_features]
ionocytes_features.extend(markers_dict['Ionocyte'])
ionocytes_features = [f for f in full_features if f in ionocytes_features]
ionocytes_features = list(sorted(set(ionocytes_features)))
# Load samples indices
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_pos_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_neg_ionocytes.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_matrix.mtx"))
adata = adata[:, [idx for idx, f in enumerate(full_features) if f in full_features]]
adata.var_names = [f.upper() for idx, f in enumerate(full_features) if f in full_features]
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
# sc.pp.scale(adata, max_value=10)
# Identify highly-variable genes.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
X = X[:, adata.var.highly_variable]
adata = adata[:, adata.var.highly_variable]
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.05, key_added="clusters")
new_cluster_names = ['Tuft', 'Ionocyte', 'Unconventional']
adata.rename_categories('clusters', new_cluster_names)
adata.X = X
# Filter markers
selected_markers_dict = {'Basal': ['KRT5', "KRT8"],
                         'Club': ["NFIA", "MUC5B", "SCGB1A1", "NOTCH2"],
                         'Tuft': ["RGS13", "ALOX5AP", "PTPRC", "POU2F3"],
                         'PNEC': ['ASCL1', 'ASCL2'],
                         'Ionocyte': ['CFTR', 'FOXI1', 'ASCL3'],
                         'Ciliated': ["FOXJ1", "CETN2", "TUBA1A", 'CDHR3'],
                         'Goblet': ['FOXQ1']}
temp_dict = {}
for key, items in selected_markers_dict.items():
    temp = []
    for item in items:
        if item in adata.var_names:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
selected_markers_dict = temp_dict

temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in adata.var_names:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict

ionocytes_features.extend(["CFTR", "FOXI1", "POU2F3", "EGR1"])
ionocytes_features = np.unique(ionocytes_features)
ionocytes_features = [item for item in ionocytes_features if item in adata.var.index]

# Plot UMAP
with plt.rc_context({'figure.figsize': (8, 5), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'],
               use_raw=False, add_outline=False, legend_loc='on data',
               frameon=False, title='Clusters', legend_fontsize=12,
               palette={"Tuft": "blue", "Ionocyte": "black", "Unconventional": "red"})
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['CFTR', 'FOXI1', 'ASCL3', 'EGR1'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2, frameon=False)
# Plot differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon')
with plt.rc_context({'figure.figsize': (8, 5), 'figure.labelsize': '20',
                     'axes.titlesize': '24', 'axes.labelsize': '20',
                     'xtick.labelsize': '14', 'ytick.labelsize': '14'}):
    sc.pl.rank_genes_groups(adata, n_genes=20, fontsize=18, sharey=False,
                            **{"axes.xlabel": "Ranking"})
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['CFTR', 'FOXI1', 'ASCL3', 'EGR1'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, ionocytes_features, groupby='clusters', title=None,
                  colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, ionocytes_features, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, swap_axes=False, figsize=(11, 8))
    sc.pl.heatmap(adata, markers_dict["Ionocyte"], groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, var_group_rotation=45, swap_axes=False,
                  figsize=(11, 6))
    sc.pl.heatmap(adata, markers_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                  dendrogram=False, var_group_rotation=45, swap_axes=False,
                  figsize=(11, 3))

##############################################################
####### Remove Pulseseq Negative Tuft & Ionocyte cells #######
##############################################################
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                       "pulseseq_pos_tuft_ionocytes.txt"),
                          sep=',', header=None)[0]
pos_samples = np.squeeze(pos_samples.values.tolist())
# Load data
X = sc.read_mtx(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_matrix.mtx"))
X = X.T[pos_samples].to_df().to_numpy()
X = coo_matrix(X)
# Save data
mmwrite(target=os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"), a=X)
df = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_classes.csv"),
                 sep=',')
columns = df.columns.to_list()
df = df.to_numpy()[pos_samples]
df = pd.DataFrame(df, columns=columns)
df.to_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
          sep=',', index=False)

df = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_donors.csv"),
                 sep=',')
columns = df.columns.to_list()
df = df.to_numpy()[pos_samples]
df = pd.DataFrame(df, columns=columns)
df.to_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_donors.csv"),
          sep=',', index=False)

df = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_timepoints.csv"),
                 sep=',')
columns = df.columns.to_list()
df = df.to_numpy()[pos_samples]
df = pd.DataFrame(df, columns=columns)
df.to_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
          sep=',', index=False)

df = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_types.csv"),
                 sep=',')
columns = df.columns.to_list()
df = df.to_numpy()[pos_samples]
df = pd.DataFrame(df, columns=columns)
df.to_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_types.csv"),
          sep=',', index=False)

##############################################################
##### Analyze Tuft vs Ionocytes without Outliers (PHet) ######
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_tuft_vs_ionocyte_exclude_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# Ionocytes markers
markers = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
ionocyte_markers = np.unique(np.squeeze(markers[["Ionocyte"]].values.tolist()).flatten())[1:]
temp = pd.read_csv(os.path.join(DATASET_PATH,
                                "cell_type_category_human_rna_ionocytes.tsv"),
                   sep='\t')["Gene"].to_list()
ionocyte_markers = np.append(ionocyte_markers, temp)
ionocyte_markers = [f.upper() for f in ionocyte_markers if f.upper() in phet_features]
ionocyte_markers = np.unique(ionocyte_markers)
# Tuft markers
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers if f.upper() in phet_features]
tuft_markers = np.unique(tuft_markers)
# Extract all pulseseq tuft and ionocytes markers
markers = np.unique(np.concatenate((ionocyte_markers, tuft_markers)))
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"],
                'Ionocyte': ['CFTR', 'FOXI1', 'ASCL3']}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in phet_features:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
tuft_samples = np.where(classes == 0)[0]
ionocyte_samples = np.where(classes == 1)[0]
samples_idx = np.append(tuft_samples, ionocyte_samples)
group_name = ["Tuft"] * len(tuft_samples) + ["Ionocyte"] * len(ionocyte_samples)

#  Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in phet_features]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in phet_features]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell, so that counts become comparable among cells.
sc.pp.normalize_total(adata, target_sum=1e4)
# Logarithmize the data:
sc.pp.log1p(adata)
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.4, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Tuft-1', 'Ionocyte', 'Tuft-2', 'Tuft-3']
adata.rename_categories('clusters', new_cluster_names)
adata.obs["clusters"][group_name == "Ionocyte"] = "Ionocyte"
temp = adata.obs["clusters"][group_name == "Tuft"]
temp[temp == "Ionocyte"] = "Tuft-3"
adata.obs["clusters"][group_name == "Tuft"] = temp
# Plot UMAP
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'], use_raw=False, add_outline=False,
               legend_loc='on data', legend_fontsize=25, legend_fontoutline=0,
               frameon=False)
    sc.pl.umap(adata, color=['clusters', 'POU2F3', 'GNAT3', 'TRPM5', 'HCK',
                             'LRMP', 'RGS13', 'GNG13', 'ALOX5AP', 'CFTR', 'FOXI1', 'ASCL3'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP', 'RGS13',
                         'GNG13', 'ALOX5AP', 'CFTR', 'FOXI1', 'ASCL3'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.rank_genes_groups_tracksplot(adata, n_genes=15, dendrogram=False,
                                       xlabel="Clusters")
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=False)
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, markers_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '15'}):
    sc.pl.heatmap(adata, tuft_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))
    # sc.pl.heatmap(adata, ionocyte_markers, groupby='clusters',
    #               layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
    #               swap_axes=False, figsize=(12, 6))
    sc.pl.heatmap(adata, markers_dict, groupby='clusters', layer='scaled', vmin=-2,
                  vmax=2, cmap='RdBu_r', dendrogram=False, swap_axes=True, figsize=(12, 3))

##############################################################
#### Analyze Tuft vs Ionocytes without Outliers (Markers) ####
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Ionocytes markers
markers = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
ionocyte_markers = np.unique(np.squeeze(markers[["Ionocyte"]].values.tolist()).flatten())[1:]
ionocyte_markers = [f.upper() for f in ionocyte_markers]
ionocyte_markers = np.unique(ionocyte_markers)
ionocyte_markers = [f for f in ionocyte_markers if f in full_features]
# Tuft markers
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers]
tuft_markers = np.unique(tuft_markers)[1:]
tuft_markers = [f for f in tuft_markers if f in full_features]
# Extract all pulseseq tuft and ionocytes markers
markers = np.unique(np.concatenate((ionocyte_markers, tuft_markers)))
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"],
                'Ionocyte': ['CFTR', 'FOXI1', 'ASCL3']}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in markers:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
tuft_samples = np.where(classes == 0)[0]
ionocyte_samples = np.where(classes == 1)[0]
samples_idx = np.append(tuft_samples, ionocyte_samples)
group_name = ["Tuft"] * len(tuft_samples) + ["Ionocyte"] * len(ionocyte_samples)

# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in markers]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in markers]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# # Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.8, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Ionocyte', 'Tuft-2', 'Tuft-1', 'Tuft-3']
adata.rename_categories('clusters', new_cluster_names)
adata.obs["clusters"][group_name == "Ionocyte"] = "Ionocyte"
temp = adata.obs["clusters"][group_name == "Tuft"]
temp[temp == "Ionocyte"] = "Tuft-1"
adata.obs["clusters"][group_name == "Tuft"] = temp
# Plot UMAP 
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters', 'POU2F3', 'GNAT3', 'TRPM5',
                             'HCK', 'LRMP', 'RGS13', 'GNG13', 'ALOX5AP', 'CFTR',
                             'FOXI1', 'ASCL3'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=2, frameon=False)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP', 'RGS13',
                         'GNG13', 'ALOX5AP', 'CFTR', 'FOXI1', 'ASCL3'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.rank_genes_groups_tracksplot(adata, n_genes=15, dendrogram=True,
                                       xlabel="Clusters")
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=True)
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, markers_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, tuft_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))
    sc.pl.heatmap(adata, ionocyte_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))

##############################################################
###### Analyze Tuft vs Ionocytes without Outliers (HVF) ######
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Ionocytes markers
markers = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
ionocyte_markers = np.unique(np.squeeze(markers[["Ionocyte"]].values.tolist()).flatten())[1:]
temp = pd.read_csv(os.path.join(DATASET_PATH,
                                "cell_type_category_human_rna_ionocytes.tsv"),
                   sep='\t')["Gene"].to_list()
ionocyte_markers = np.append(ionocyte_markers, temp)
ionocyte_markers = [f.upper() for f in ionocyte_markers if f.upper() in full_features]
ionocyte_markers = np.unique(ionocyte_markers)
# Tuft markers
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers if f.upper() in full_features]
tuft_markers = np.unique(tuft_markers)
# Extract all pulseseq tuft and ionocytes markers
markers = np.unique(np.concatenate((ionocyte_markers, tuft_markers)))
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"],
                'Ionocyte': ['CFTR', 'FOXI1', 'ASCL3']}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in full_features:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
tuft_samples = np.where(classes == 0)[0]
ionocyte_samples = np.where(classes == 1)[0]
samples_idx = np.append(tuft_samples, ionocyte_samples)
group_name = ["Tuft"] * len(tuft_samples) + ["Ionocyte"] * len(ionocyte_samples)

# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx]
adata.var_names = full_features
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Identify highly-variable genes.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
X = X[:, adata.var.highly_variable]
adata = adata[:, adata.var.highly_variable]
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in adata.var_names:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
tuft_markers = [item for item in tuft_markers if item in adata.var_names]
ionocyte_markers = [item for item in ionocyte_markers if item in adata.var_names]
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.4, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Tuft-2', 'Tuft-1', 'Ionocyte', 'Tuft-3']
adata.rename_categories('clusters', new_cluster_names)
adata.obs["clusters"][group_name == "Ionocyte"] = "Ionocyte"
temp = adata.obs["clusters"][group_name == "Tuft"]
temp[temp == "Ionocyte"] = "Tuft-3"
adata.obs["clusters"][group_name == "Tuft"] = temp
# Plot UMAP
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters', 'HCK',
                             'RGS13', 'ALOX5AP', 'CFTR', 'FOXI1', 'ASCL3'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=2, frameon=False)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['HCK', 'RGS13', 'ALOX5AP', 'CFTR', 'FOXI1', 'ASCL3'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.rank_genes_groups_tracksplot(adata, n_genes=15, dendrogram=True,
                                       xlabel="Clusters")
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=True)
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, markers_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, tuft_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))
    sc.pl.heatmap(adata, ionocyte_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))
    sc.pl.heatmap(adata, markers_dict, groupby='clusters', layer='scaled', vmin=-2,
                  vmax=2, cmap='RdBu_r', dendrogram=True, swap_axes=True, figsize=(12, 3))

##############################################################
################  Analyze Tuft Cells (PHet)  #################
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "pulseseq",
                                         "pulseseq_tuft_vs_ionocyte_exclude_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# Tuft markers
markers = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers if f.upper() in phet_features]
tuft_markers = np.unique(tuft_markers)
# Extract all pulseseq tuft markers
markers = np.unique(tuft_markers)
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"]}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in phet_features:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
samples_idx = np.where(classes == 0)[0]
group_name = ["Tuft"] * len(samples_idx)
timepoints = timepoints[samples_idx]

# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in phet_features]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in phet_features]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.5, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Tuft-1', 'Tuft-2', 'Tuft-3', 'Tuft-4']
adata.rename_categories('clusters', new_cluster_names)
phet_scores = list()
le = LabelEncoder()
subtypes = le.fit_transform(adata.obs["clusters"])
phet_scores.extend(clustering_performance_wo_ground_truth(X=adata.obsm["X_umap"], labels_pred=subtypes))
G = nx.from_numpy_array(adata.obsp["connectivities"])
mod = nx.community.modularity(G, [list(np.where(adata.obs['clusters'] == 'Tuft-1')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-2')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-3')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-4')[0])])
phet_scores.append(mod)
# Plot UMAP
tuft_palette = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
adata.uns["clusters_colors"] = tuft_palette
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters', 'POU2F3', 'GNAT3', 'TRPM5', 'HCK',
                             'LRMP', 'FOXE1', 'RGS2', 'RGS13', 'GNG13', 'ALOX5AP', 'ALOX5'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP', 'RGS13',
                         'GNG13', 'ALOX5AP'],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=False)
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, markers_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, tuft_markers, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(12, 6))
    sc.pl.heatmap(adata, markers_dict, groupby='clusters', layer='scaled', vmin=-2,
                  vmax=2, cmap='RdBu_r', dendrogram=False, swap_axes=True,
                  figsize=(12, 3))

##############################################################
##############   Analyze Tuft Cells (Markers)  ###############
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Tuft markers
markers = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers]
tuft_markers = np.unique(tuft_markers)[1:]
tuft_markers = [f for f in tuft_markers if f in full_features]
# Extract all pulseseq tuft and ionocytes markers
markers = np.unique(tuft_markers)
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"]}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in markers:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
samples_idx = np.where(classes == 0)[0]
group_name = ["Tuft"] * len(samples_idx)
timepoints = timepoints[samples_idx]

# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features) if f in markers]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in markers]
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.8, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Tuft-2', 'Tuft-1', 'Tuft-3', 'Tuft-4']
adata.rename_categories('clusters', new_cluster_names)
markers_scores = list()
le = LabelEncoder()
subtypes = le.fit_transform(adata.obs["clusters"])
markers_scores.extend(clustering_performance_wo_ground_truth(X=adata.obsm["X_umap"], labels_pred=subtypes))
G = nx.from_numpy_array(adata.obsp["connectivities"])
mod = nx.community.modularity(G, [list(np.where(adata.obs['clusters'] == 'Tuft-1')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-2')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-3')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-4')[0])])
markers_scores.append(mod)
# Plot UMAP
tuft_palette = ['#dd8452', '#4c72b0', '#55a868', '#c44e52']
adata.uns["clusters_colors"] = tuft_palette
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters', 'HCK', 'RGS13', 'ALOX5AP'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, markers_dict["Tuft"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=False)

##############################################################
################   Analyze Tuft Cells (HVF)  #################
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "pulseseq_tuft_vs_ionocyte_exclude_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Tuft markers
markers = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_markers.csv"), sep=',').replace(np.nan, -1)
tuft_markers = np.squeeze(markers[["Tuft", "Tuft-1", "Tuft-2"]].values.tolist()).flatten()[1:]
tuft_markers = [f.upper() for f in tuft_markers if f.upper() in full_features]
tuft_markers = np.unique(tuft_markers)
# Extract all pulseseq tuft markers
markers = np.unique(tuft_markers)
markers = [f for f in markers if f in full_features]
# selected markers
markers_dict = {'Tuft': ['POU2F3', 'GNAT3', 'TRPM5', 'HCK', 'LRMP',
                         'RGS13', 'GNG13', 'ALOX5AP', "PTPRC"]}
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in full_features:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
del temp, temp_dict
# timelabels
timepoints = pd.read_csv(os.path.join(DATASET_PATH, "pulseseq_tuft_vs_ionocyte_exclude_timepoints.csv"),
                         sep=',').values.tolist()
timepoints = np.squeeze(timepoints)
timepoints = pd.Series(timepoints, dtype="category")
# Classes
classes = pd.read_csv(os.path.join(DATASET_PATH,
                                   "pulseseq_tuft_vs_ionocyte_exclude_classes.csv"),
                      sep=',').values.tolist()
classes = np.squeeze(classes)
samples_idx = np.where(classes == 0)[0]
group_name = ["Tuft"] * len(samples_idx)
timepoints = timepoints[samples_idx]
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH,
                                 "pulseseq_tuft_vs_ionocyte_exclude_matrix.mtx"))
adata = adata[samples_idx]
adata.var_names = full_features
group_name = pd.Series(group_name, dtype="category")
group_name.index = adata.obs.index
adata.obs["clusters"] = group_name
timepoints.index = adata.obs.index
adata.obs["timepoints"] = timepoints
X = copy.deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Identify highly-variable genes.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
X = X[:, adata.var.highly_variable]
adata = adata[:, adata.var.highly_variable]
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item in adata.var_names:
            temp.append(item)
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
tuft_markers = [item for item in tuft_markers if item in adata.var_names]
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.6, key_added="clusters")
# Rename clusters
adata.X = X
new_cluster_names = ['Tuft-2', 'Tuft-3', 'Tuft-1', 'Tuft-4']
adata.rename_categories('clusters', new_cluster_names)
hvf_scores = list()
le = LabelEncoder()
subtypes = le.fit_transform(adata.obs["clusters"])
hvf_scores.extend(clustering_performance_wo_ground_truth(X=adata.obsm["X_umap"], labels_pred=subtypes))
G = nx.from_numpy_array(adata.obsp["connectivities"])
mod = nx.community.modularity(G, [list(np.where(adata.obs['clusters'] == 'Tuft-1')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-2')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-3')[0]),
                                  list(np.where(adata.obs['clusters'] == 'Tuft-4')[0])])
hvf_scores.append(mod)
# Plot UMAP
tuft_palette = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
tuft_palette = ['#dd8452', '#55a868', '#4c72b0', '#c44e52']
adata.uns["clusters_colors"] = tuft_palette
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters', 'HCK', 'RGS13', 'ALOX5AP'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Violin plot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, markers_dict["Tuft"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Tracksplot
with plt.rc_context({'figure.figsize': (12, 6), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '20',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.tracksplot(adata, markers_dict, groupby='clusters',
                     dendrogram=False)

### Gather all scores
df_scores = pd.DataFrame([phet_scores, markers_scores, hvf_scores]).T
df_scores.columns = ["PHet", "Markers", "Dispersion"]
df_scores.index = ["Complete Diameter Distance", "Average Diameter Distance",
                   "Centroid Diameter Distance", "Single Linkage Distance",
                   "Maximum Linkage Distance", "Average Linkage Distance",
                   "Centroid Linkage Distance", "Ward's Distance", "Silhouette",
                   "Calinski-Harabasz", "Davies-Bouldin", "Modularity"]
df_scores.to_csv(path_or_buf=os.path.join(RESULT_PATH, "tuft_cluster_quality.csv"),
                 sep=",")
