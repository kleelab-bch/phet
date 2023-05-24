import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scanpy as sc
import seaborn as sns
from copy import deepcopy

from utility.file_path import RESULT_PATH, DATASET_PATH

sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')
sns.set_theme()
sns.set_style(style='white')

##############################################################
######################## Plasschaert #########################
##############################################################
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                              "plasschaert_mouse_groups.csv"),
                 sep=',', index_col=0, header=None)
df = df.T
temp = {"secretory": "Secretory", "basal": "Basal", "ciliated": "Ciliated",
        "brush": "Brush", "pnec": "PNEC", "krt4/13+": "KRT4/13+",
        "cycling basal (homeostasis)": "Cycling basal (homeostasis)",
        "ionocytes": "Ionocyte",
        "cycling basal (regeneration)": "Cycling basal (regeneration)",
        "pre-ciliated": "Pre-ciliated"}
temp = [temp[item] for item in df["subtypes"].tolist()]
df["subtypes"] = temp

# Use static colors
palette = {"Secretory": "#1f77b4", "Basal": "#ff7f0e", "Ciliated": "#2ca02c",
           "Brush": "#d62728", "PNEC": "#9467bd", "KRT4/13+": "#8c564b",
           "Cycling basal (homeostasis)": "#e377c2", "Ionocyte": "#7f7f7f",
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

# manually generate legend
plt.figure(figsize=(4, 2))
ax = sns.barplot(data=distribution.iloc[:, ::-1].cumsum(axis=1)
                 .stack().reset_index(name='Dist'),
                 x='timepoints', y='Dist', hue='subtypes',
                 hue_order=distribution.columns, width=0.9,
                 dodge=False, palette=palette)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.legend(title=None, fontsize=30, ncol=2, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
sns.despine()
plt.tight_layout()

#### load donors features data
top_down_features = 25
ciliated_features = pd.read_csv(os.path.join(DATASET_PATH,
                                             "cell_type_category_human_rna_ciliated.tsv"),
                                sep='\t')
ciliated_features = ciliated_features["Gene"].to_list()
ciliated_features = [item.lower() for item in ciliated_features]
features = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                    "plasschaert_mouse_phet_b_features.csv"),
                       sep=',', header=None)
features = np.squeeze(features.values.tolist())
temp = [f for f in features if f.lower() in ciliated_features]
temp.extend(["Msln"])

enriched_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                        "enriched_terms_uninjured_vs_injured.txt"),
                           sep='\t', header=None)
enriched_idx.columns = ["Features", "Scores"]
enriched_idx = enriched_idx["Features"].to_list()

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
enriched_idx = np.unique([idx for idx, f in enumerate(features) if f in temp])
features = features[enriched_idx]

# load donors data
pos_samples = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "uninjured.txt"),
                          sep=',', index_col=0, header=None)
pos_samples = np.squeeze(pos_samples.values.tolist())
neg_samples = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "injured.txt"),
                          sep=',', index_col=0, header=None)
neg_samples = np.squeeze(neg_samples.values.tolist())
samples_idx = np.append(pos_samples, neg_samples)
samples_name = ["Uninjured"] * len(pos_samples) + ["Injured"] * len(neg_samples)
samples_name = pd.Series(samples_name)
lut = dict(zip(samples_name.unique(), ["black", "red"]))
row_colors = list(samples_name.map(lut))

# load expression
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                              "plasschaert_mouse_phet_b_expression.csv"),
                 sep=',', header=None)
df = df.iloc[samples_idx, enriched_idx]
df.metrics_name = features

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
############# Plasschaert Top vs Bottom (PHet) ###############
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "plasschaert_mouse_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                         "plasschaert_mouse_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# load markers
ciliated_features = pd.read_csv(os.path.join(DATASET_PATH,
                                             "cell_type_category_human_rna_ciliated.tsv"),
                                sep='\t')
ciliated_features = ciliated_features["Gene"].to_list()
ciliated_features = [item.upper() for item in ciliated_features]
df = pd.read_csv(os.path.join(DATASET_PATH,
                              "plasschaert_mouse_all_markers.csv"),
                 sep=',', header=0)
df.columns = ["Genes", "Cells"]
markers_dict = {}
for cell in np.unique(df["Cells"]):
    temp = df["Cells"] == cell
    if cell in ['PNEC', 'Ionocytes', 'Brush']:
        continue
    if cell in ['Basal', 'Basal & Cycling basal pooled']:
        cell = 'Basal'
    temp = sorted([f.upper() for f in df[temp]["Genes"].values.tolist()
                   if not f.lower().startswith("rp")])
    temp = list(set(temp))
    if cell.capitalize() in markers_dict.keys():
        markers_dict[cell.capitalize()] = temp + markers_dict[cell]
    else:
        markers_dict[cell.capitalize()] = temp
del df
markers_dict["Ciliated"] += ciliated_features
markers_dict["Ciliated"] = list(set(markers_dict["Ciliated"]))
markers = np.unique([i for k, item in markers_dict.items()
                     for i in item])
# load trajectory data
top_cluster = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "plasschaert_mouse_trajectory_top.txt"),
                          sep=',', index_col=0, header=None)
top_cluster = np.squeeze(top_cluster.values.tolist())
bottom_cluster = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                          "plasschaert_mouse_trajectory_bottom.txt"),
                             sep=',', index_col=0, header=None)
bottom_cluster = np.squeeze(bottom_cluster.values.tolist())
samples_idx = np.append(top_cluster, bottom_cluster)
samples_name = ["Top cluster"] * len(top_cluster) + ["Bottom cluster"] * len(bottom_cluster)
samples_name = pd.Series(samples_name)
lut = dict(zip(samples_name.unique(), ["black", "red"]))
row_colors = list(samples_name.map(lut))
# load enriched featrues
top_down_features = 25
enriched_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                        "enriched_terms_bottom_vs_top.txt"),
                           sep='\t', header=None)
enriched_idx.columns = ["Features", "Scores"]
enriched_idx = [f.upper() for f in enriched_idx["Features"].to_list()]
temp = []
to_add = (2 * top_down_features - len(temp)) // 2 + len(temp)
while len(temp) < to_add:
    if len(enriched_idx) == 0:
        break
    f = enriched_idx.pop(0)
    if f.lower().startswith("rp"):
        continue
    if f not in markers:
        continue
    temp.append(f)
while len(temp) < top_down_features * 2:
    if len(enriched_idx) == 0:
        break
    f = enriched_idx.pop(-1)
    if f.lower().startswith("rp"):
        continue
    if f not in markers:
        continue
    temp.append(f)
enriched_idx = np.unique([idx for idx, f in enumerate(phet_features) if f in temp])
heat_features = np.array(phet_features)[enriched_idx]
#  load expression
df = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                              "plasschaert_mouse_phet_b_expression.csv"),
                 sep=',', header=None)
df = df.iloc[samples_idx, enriched_idx]
df.metrics_name = heat_features
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
del df, cg, ax

# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH, "plasschaert_mouse_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features)
                               if f in markers]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in markers]
samples_name = pd.Series(samples_name, dtype="category")
samples_name.index = adata.obs.index
adata.obs["clusters"] = samples_name
X = deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Regress out effects of total counts per cell 
sc.pp.regress_out(adata, ['total_counts'])
# Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)
# Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
# UMAP (Embedding the neighborhood graph)
sc.tl.umap(adata, min_dist=0.0, spread=1.0, n_components=2,
           maxiter=2000)
adata.X = X
sc.pp.scale(adata, max_value=10)
# Plot UMAP
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=40, legend_fontoutline=2, frameon=False,
               palette={"Bottom cluster": "black", "Top cluster": "red"})
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Plot ranked genes
with plt.rc_context({'figure.figsize': (8, 6), 'figure.labelsize': '20',
                     'axes.titlesize': '24', 'axes.labelsize': '20',
                     'xtick.labelsize': '14', 'ytick.labelsize': '14'}):
    sc.pl.rank_genes_groups(adata, n_genes=20, fontsize=18, sharey=False,
                            **{"axes.xlabel": "Ranking"})
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
markers = [f for k, item in markers_dict.items() for f in item]
selected_features_dict = {'Basal': ['KRT5', 'TRP63'],
                          'KRT4/13': ['KRT13', 'Krt4'],
                          'Secretory': ['MUC5B', 'SCGB1A1', 'NOTCH2', 'BPIFA1', 'MSLN', 'AGR2'],
                          'Ciliated': ['FOXJ1', 'MYB', 'BASP1', 'PTGES3', 'CETN2',
                                       'TUBA1A', 'CDHR3'],
                          'Cycling basal': ['SPRR2A2']}
temp_dict = {}
for key, items in selected_features_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
selected_features_dict = temp_dict
selected_features = [f for k, item in selected_features_dict.items() for f in item]
# Violin plot
adata.var_names_make_unique()
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, selected_features_dict["Basal"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, selected_features_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
    selected_samples = adata.obs['clusters'] == "Top cluster"
    mean_expressions = adata[selected_samples, markers_dict["Ciliated"]].to_df().median()
    mean_expressions = np.where(mean_expressions > 0.0)[0]
    sc.pl.dotplot(adata, np.array(markers_dict["Ciliated"])[mean_expressions],
                  groupby='clusters', title=None,
                  colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
    mean_expressions = adata[selected_samples, markers_dict["Secretory"]].to_df().median()
    mean_expressions = np.where(mean_expressions > 0.0)[0]
    sc.pl.dotplot(adata, np.array(markers_dict["Secretory"])[mean_expressions],
                  groupby='clusters', title=None,
                  colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
    mean_expressions = adata[selected_samples, markers_dict["Cycling basal"]].to_df().median()
    mean_expressions = np.where(mean_expressions > 0.0)[0]
    sc.pl.dotplot(adata, np.array(markers_dict["Cycling basal"])[mean_expressions],
                  groupby='clusters', title=None,
                  colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")

# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '8'}):
    sc.pl.heatmap(adata, markers_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(10, 2))
    sc.pl.heatmap(adata, selected_features_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(6, 1.5))
# Cell populations
adata = sc.read_mtx(os.path.join(DATASET_PATH, "plasschaert_mouse_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features)
                               if f in markers]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in markers]
samples_name = pd.Series(samples_name, dtype="category")
samples_name.index = adata.obs.index
adata.obs["clusters"] = samples_name
# Detect cell populations
basal_markers = ["KRT5", "TRP63", "KRT14", "PDPN", "NGFR", "LGALS1", "ITGA6",
                 "ITGB4", "LAMA3", "LAMB3", "KRT15", "S100A2", "NPPC", "BCAM",
                 "DST", "KRT19", "NOTCH3", "DCN", "COL17A1", "ABI3BP"]
basal_markers = [f for f in basal_markers if f in markers_dict["Basal"]]
# selected_features_dict = {'Basal': ['KRT5', 'TRP63'], 
#                           'KRT4/13+': ['KRT4', 'KRT13'],
#                           'Secretory': marker_features_dict["Secretory"],
#                           'Ciliated':  marker_features_dict["Ciliated"],
#                           'Cycling basal': marker_features_dict['Cycling basal']}
selected_features_dict = {'Basal': basal_markers,
                          'KRT4/13+': markers_dict["Krt4/13+"],
                          'Secretory': markers_dict["Secretory"],
                          'Ciliated': markers_dict["Ciliated"],
                          'Cycling basal': markers_dict['Cycling basal']}
cell_populations = np.zeros((len(selected_features_dict.keys()), 2))
cell_types = {}
for cluster_idx, cluster in enumerate(["Top cluster", "Bottom cluster"]):
    selected_samples = adata.obs['clusters'] == cluster
    adata_copy = adata[selected_samples]
    sc.pp.calculate_qc_metrics(adata_copy, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.scale(adata_copy, max_value=1)
    exclude_samples = []
    for cell_idx, cell in enumerate(list(selected_features_dict.keys())):
        temp = []
        cells_per_feature = []
        # for f in selected_features_dict[cell]:
        #     idx = np.where(adata_copy.var_names == f)[0]
        #     samples_idx = np.where(adata_copy.X[:, idx] > 0)[0]
        #     samples_idx = np.array([i for i in samples_idx if i not in exclude_samples])
        #     if samples_idx.shape[0] == 0:
        #         continue
        #     majority_f = np.percentile(np.sort(adata.X[samples_idx][:, idx].todense())[::-1], 90)
        #     temp_idx = samples_idx[np.where(adata.X[samples_idx][:, idx].todense() > majority_f)[0]]
        #     temp.append(temp_idx.shape[0])
        #     cells_per_feature.extend(temp_idx)
        # if len(cells_per_feature) == 0:
        #     continue
        # cell_types[cell] = list(set(cells_per_feature))
        # exclude_samples.extend(cell_types[cell])
        # temp = np.mean(temp)
        temp = adata_copy.var["mean_counts"][selected_features_dict[cell]]
        temp = np.percentile(np.sort(temp), 50)
        cell_populations[cell_idx, cluster_idx] = temp
cell_populations[cell_populations < 0] = 0
cell_populations /= cell_populations.sum(0)
cell_populations = pd.DataFrame(cell_populations, columns=["Top cluster", "Bottom cluster"],
                                index=selected_features_dict.keys())
cell_populations = cell_populations.T
cell_populations.index.names = ["Cluster"]
cell_populations.reset_index(inplace=True)

# Use static colors
palette = {"Basal": "#ff7f0e", "KRT4/13+": "#1f77b4", "Secretory": "#8c564b", "Ciliated": "#2ca02c",
           "Cycling basal": "#d62728"}
plt.figure(figsize=(6, 8))
ax = cell_populations.plot(kind='bar', stacked=True, width=0.9, color=palette)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=16)
ax.set_xticklabels(["Top cluster", "Bottom cluster"], rotation=45, fontsize=20)
ax.set_ylabel("Percentage of all cells", fontsize=20)
ax.set_xlabel(None)
ax.legend(title=None, fontsize=26, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
plt.tight_layout()

##############################################################
################## Plasschaert Basal (PHet) ##################
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "plasschaert_mouse_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                         "plasschaert_mouse_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# load basal subsets markers
df = pd.read_csv(os.path.join(DATASET_PATH, "basal_subsets_markers.csv"),
                 sep=',', header=0)
df = df[["gene", "cluster", "Signature"]]
df = df[df["Signature"] == "yes"]
basal_subsets_markers = {}
for cluster in set(df["cluster"]):
    if cluster == "Basal1":
        name_cluster = "Canonical basal"
    elif cluster == "Basal2":
        name_cluster = "Proliferating basal"
    elif cluster == "Basal3":
        name_cluster = "Serpins expressing basal"
    elif cluster == "Basal4":
        name_cluster = "JUN/FOS"
    else:
        name_cluster = "Beta-catenin"
    temp = df[df["cluster"] == cluster]["gene"].values.tolist()
    temp = [f for f in temp if f in full_features]
    basal_subsets_markers[name_cluster] = temp
# load all markers
df = pd.read_csv(os.path.join(DATASET_PATH,
                              "plasschaert_mouse_all_markers.csv"),
                 sep=',', header=0)
df.columns = ["Genes", "Cells"]
markers_dict = {}
for cell in np.unique(df["Cells"]):
    temp = df["Cells"] == cell
    if cell in ['PNEC', 'Ionocytes', 'Brush', 'Ciliated', 'Cycling Basal']:
        continue
    if cell in ['Basal', 'Basal & Cycling basal pooled']:
        cell = 'Basal'
    temp = sorted([f.upper() for f in df[temp]["Genes"].values.tolist()
                   if not f.lower().startswith("rp")])
    temp = list(set(temp))
    if cell.capitalize() in markers_dict.keys():
        markers_dict[cell.capitalize()] = temp + markers_dict[cell]
    else:
        markers_dict[cell.capitalize()] = temp
del df
markers = np.unique([i for k, item in markers_dict.items()
                     for i in item])
# load trajectory data
samples_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "plasschaert_mouse_basal.txt"),
                          sep=',', index_col=0, header=None)
samples_idx = np.squeeze(samples_idx.values.tolist())
# load donors data
donors = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                  "plasschaert_mouse_groups.csv"),
                     sep=',', index_col=0, header=None)
donors = donors.T["timepoints"]
donors = ["Injured" if item != "uninjured" else "Uninjured"
          for item in donors.values.tolist()]
donors = pd.Series(donors, dtype="category")
donors = donors[samples_idx]
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH, "plasschaert_mouse_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features)
                               if f in phet_features]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in phet_features]
donors.index = adata.obs.index
adata.obs["donors"] = donors
X = deepcopy(adata.X)
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
sc.tl.leiden(adata, resolution=0.5, key_added="clusters")
adata.obs["clusters"] = ["3" if item == "4" else item for item in adata.obs['clusters']]
adata.obs["clusters"] = pd.Series(adata.obs["clusters"], name="clusters", dtype="category")
# Rename clusters
new_cluster_names = ['Basal-1', 'Basal-2', 'Basal-3', 'Basal-4']
adata.rename_categories('clusters', new_cluster_names)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
markers = [f for k, item in markers_dict.items() for f in item]
basal_markers = ["KRT5", "KRT14", "KRT15", "KRT19", "TRP63", "PDPN", "NGFR",
                 "ITGA6", "ITGB4", "LAMB3", "BCAM", "DST", "DCN", "COL17A1",
                 "ABI3BP", "TSPAN1", "WFDC2", "ATP1B1", "DUT", "CXCL17"]
basal_markers += [f.upper() for item in basal_subsets_markers.values() for f in item]
basal_markers = sorted(list(set(basal_markers)))
secretory_markers = ["MUC5AC", "MUC5B", "TFF3", "SCGB3A1", "SCGB3A2",
                     "BPIFB1", "MSMB", "SLPI", "WFDC2", "BPIFA1", "MSLN", "AGR2"]
secretory_markers = sorted(list(set(secretory_markers)))
selected_features_dict = {'Basal': ['COL17A1', 'KRT15', 'KRT5', 'LAMB3', 'TRP63',
                                    'TSPAN1'],
                          'Secretory': ['AGR2', 'BPIFA1', 'MSLN', 'MUC5B',
                                        'SCGB1A1', 'SCGB3A2']}
selected_features_dict = {'Basal': basal_markers,
                          'Secretory': secretory_markers}
temp_dict = {}
for key, items in selected_features_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
selected_features_dict = temp_dict
selected_features = [f for k, item in selected_features_dict.items() for f in item]
temp_dict = {}
for key, items in basal_subsets_markers.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
basal_subsets_markers = temp_dict
# Plot UMAP 
adata.uns["donors_colors"] = ["#F05454", "#59CE8F"]
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'] + ['donors'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
    sc.pl.umap(adata, color=['clusters'] + ['donors'] + selected_features,
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Plot distributions of basal cells in injury vs uninjury conditions
basal_populations = list()
for d in sorted(set(adata.obs["donors"])):
    df = adata[adata.obs["donors"] == d].obs.groupby(["clusters"]).count()["donors"]
    df /= df.sum()
    temp = df.to_list()
    temp.insert(0, d)
    basal_populations.append(temp)
basal_populations = pd.DataFrame(basal_populations, columns=["Cluster"] + df.index.to_list())
# Use static colors
basal_palette = {"Basal-1": "#4c72b0", "Basal-2": "#dd8452",
                 "Basal-3": "#55a868", "Basal-4": "#c44e52"}
plt.figure(figsize=(10, 8))
ax = basal_populations.plot(kind='bar', stacked=True, width=0.9,
                            color=basal_palette)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=16)
ax.set_xticklabels(["Injured", "Uninjured"], rotation=45, fontsize=20)
ax.set_ylabel("Percentage of basal cells", fontsize=20)
ax.set_xlabel(None)
ax.legend(title=None, fontsize=26, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
# Violin plot
adata.X = X
adata.var_names_make_unique()
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, selected_features_dict["Basal"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
    sc.pl.violin(adata, selected_features_dict["Secretory"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
sc.pp.scale(adata)
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, basal_subsets_markers, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
    sc.pl.dotplot(adata, selected_features_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, selected_features_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(10, 2))

##############################################################
################ Plasschaert Basal (Markers) #################
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "plasschaert_mouse_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# load basal subsets markers
df = pd.read_csv(os.path.join(DATASET_PATH, "basal_subsets_markers.csv"),
                 sep=',', header=0)
df = df[["gene", "cluster", "Signature"]]
df = df[df["Signature"] == "yes"]
basal_subsets_markers = {}
for cluster in set(df["cluster"]):
    if cluster == "Basal1":
        name_cluster = "Canonical basal"
    elif cluster == "Basal2":
        name_cluster = "Proliferating basal"
    elif cluster == "Basal3":
        name_cluster = "Serpins expressing basal"
    elif cluster == "Basal4":
        name_cluster = "JUN/FOS"
    else:
        name_cluster = "Beta-catenin"
    temp = df[df["cluster"] == cluster]["gene"].values.tolist()
    temp = [f for f in temp if f in full_features]
    basal_subsets_markers[name_cluster] = temp
# load all markers
df = pd.read_csv(os.path.join(DATASET_PATH,
                              "plasschaert_mouse_all_markers.csv"),
                 sep=',', header=0)
df.columns = ["Genes", "Cells"]
markers_dict = {}
for cell in np.unique(df["Cells"]):
    temp = df["Cells"] == cell
    if cell in ['PNEC', 'Ionocytes', 'Brush', 'Ciliated', 'Cycling Basal']:
        continue
    if cell in ['Basal', 'Basal & Cycling basal pooled']:
        cell = 'Basal'
    temp = sorted([f.upper() for f in df[temp]["Genes"].values.tolist()
                   if not f.lower().startswith("rp")])
    temp = list(set(temp))
    if cell.capitalize() in markers_dict.keys():
        markers_dict[cell.capitalize()] = temp + markers_dict[cell]
    else:
        markers_dict[cell.capitalize()] = temp
del df
markers = np.unique([i for k, item in markers_dict.items()
                     for i in item])
# load trajectory data
samples_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "plasschaert_mouse_basal.txt"),
                          sep=',', index_col=0, header=None)
samples_idx = np.squeeze(samples_idx.values.tolist())
# load donors data
donors = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                  "plasschaert_mouse_groups.csv"),
                     sep=',', index_col=0, header=None)
donors = donors.T["timepoints"]
donors = ["Injured" if item != "uninjured" else "Uninjured"
          for item in donors.values.tolist()]
donors = pd.Series(donors, dtype="category")
donors = donors[samples_idx]
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH, "plasschaert_mouse_matrix.mtx"))
adata = adata[samples_idx][:, [idx for idx, f in enumerate(full_features)
                               if f in markers]]
adata.var_names = [f for idx, f in enumerate(full_features) if f in markers]
donors.index = adata.obs.index
adata.obs["donors"] = donors
X = deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
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
sc.tl.leiden(adata, resolution=0.5, key_added="clusters")
# Rename clusters
new_cluster_names = ['Basal-1', 'Basal-2', 'Basal-4', 'Basal-3']
adata.rename_categories('clusters', new_cluster_names)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
markers = [f for k, item in markers_dict.items() for f in item]
basal_markers = ['ABI3BP', 'ATP1B1', 'BCAM', 'COL17A1', 'CXCL17', 'DCN', 'DST', 'DUT',
                 'ITGA6', 'ITGB4', 'KRT15', 'KRT5', 'LAMB3', 'NGFR', 'PDPN', 'TRP63', 'TSPAN1', 'WFDC2']
basal_markers = sorted(list(set(basal_markers)))
secretory_markers = ['AGR2', 'BPIFA1', 'BPIFB1', 'MSLN', 'MUC5B', 'SCGB3A1',
                     'SCGB3A2', 'WFDC2']
secretory_markers = sorted(list(set(secretory_markers)))
selected_features_dict = {'Basal': basal_markers,
                          'Secretory': secretory_markers}
selected_features_dict = {'Basal': ['COL17A1', 'KRT15', 'KRT5', 'LAMB3', 'TRP63',
                                    'TSPAN1'],
                          'Secretory': ['AGR2', 'BPIFA1', 'MSLN', 'MUC5B',
                                        'SCGB1A1', 'SCGB3A2']}
temp_dict = {}
for key, items in selected_features_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
selected_features_dict = temp_dict
selected_features = [f for k, item in selected_features_dict.items() for f in item]
temp_dict = {}
for key, items in basal_subsets_markers.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
basal_subsets_markers = temp_dict
# Plot UMAP
adata.uns["donors_colors"] = ["#F05454", "#59CE8F"]
adata.uns["clusters_colors"] = ['#4c72b0', '#dd8452', '#c44e52', '#55a868']
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'] + ['donors'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Plot distributions of basal cells in injury vs uninjury conditions
basal_populations = list()
for d in sorted(set(adata.obs["donors"])):
    df = adata[adata.obs["donors"] == d].obs.groupby(["clusters"]).count()["donors"]
    df /= df.sum()
    temp = df.to_list()
    temp.insert(0, d)
    basal_populations.append(temp)
basal_populations = pd.DataFrame(basal_populations, columns=["Cluster"] + df.index.to_list())
# Use static colors
basal_palette = {"Basal-1": "#4c72b0", "Basal-2": "#dd8452",
                 "Basal-3": "#55a868", "Basal-4": "#c44e52"}
plt.figure(figsize=(10, 8))
ax = basal_populations.plot(kind='bar', stacked=True, width=0.9, color=basal_palette)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=16)
ax.set_xticklabels(["Injured", "Uninjured"], rotation=45, fontsize=20)
ax.set_ylabel("Percentage of basal cells", fontsize=20)
ax.set_xlabel(None)
ax.legend(title=None, fontsize=26, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
# Violin plot
adata.X = X
adata.var_names_make_unique()
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, selected_features_dict["Basal"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
    sc.pl.violin(adata, selected_features_dict["Secretory"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
sc.pp.scale(adata, max_value=10)
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, selected_features_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, selected_features_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(10, 2))

##############################################################
################### Plasschaert Basal (HVF) ##################
##############################################################
full_features = pd.read_csv(os.path.join(DATASET_PATH,
                                         "plasschaert_mouse_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
phet_features = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                         "plasschaert_mouse_phet_b_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# load basal subsets markers
df = pd.read_csv(os.path.join(DATASET_PATH, "basal_subsets_markers.csv"),
                 sep=',', header=0)
df = df[["gene", "cluster", "Signature"]]
df = df[df["Signature"] == "yes"]
basal_subsets_markers = {}
for cluster in set(df["cluster"]):
    if cluster == "Basal1":
        name_cluster = "Canonical basal"
    elif cluster == "Basal2":
        name_cluster = "Proliferating basal"
    elif cluster == "Basal3":
        name_cluster = "Serpins expressing basal"
    elif cluster == "Basal4":
        name_cluster = "JUN/FOS"
    else:
        name_cluster = "Beta-catenin"
    temp = df[df["cluster"] == cluster]["gene"].values.tolist()
    temp = [f for f in temp if f in full_features]
    basal_subsets_markers[name_cluster] = temp
# load all markers
df = pd.read_csv(os.path.join(DATASET_PATH,
                              "plasschaert_mouse_all_markers.csv"),
                 sep=',', header=0)
df.columns = ["Genes", "Cells"]
markers_dict = {}
for cell in np.unique(df["Cells"]):
    temp = df["Cells"] == cell
    if cell in ['PNEC', 'Ionocytes', 'Brush', 'Ciliated', 'Cycling Basal']:
        continue
    if cell in ['Basal', 'Basal & Cycling basal pooled']:
        cell = 'Basal'
    temp = sorted([f.upper() for f in df[temp]["Genes"].values.tolist()
                   if not f.lower().startswith("rp")])
    temp = list(set(temp))
    if cell.capitalize() in markers_dict.keys():
        markers_dict[cell.capitalize()] = temp + markers_dict[cell]
    else:
        markers_dict[cell.capitalize()] = temp
del df
markers = np.unique([i for k, item in markers_dict.items()
                     for i in item])
# load trajectory data
samples_idx = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                       "plasschaert_mouse_basal.txt"),
                          sep=',', index_col=0, header=None)
samples_idx = np.squeeze(samples_idx.values.tolist())
# load donors data
donors = pd.read_csv(os.path.join(RESULT_PATH, "plasschaert_mouse",
                                  "plasschaert_mouse_groups.csv"),
                     sep=',', index_col=0, header=None)
donors = donors.T["timepoints"]
donors = ["Injured" if item != "uninjured" else "Uninjured"
          for item in donors.values.tolist()]
donors = pd.Series(donors, dtype="category")
donors = donors[samples_idx]
# Load data
adata = sc.read_mtx(os.path.join(DATASET_PATH, "plasschaert_mouse_matrix.mtx"))
adata = adata[samples_idx]
adata.var_names = full_features
donors.index = adata.obs.index
adata.obs["donors"] = donors
X = deepcopy(adata.X)
# QC calculations
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# Total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell, so that counts become comparable among cells.
sc.pp.normalize_total(adata, target_sum=1e4)
# Logarithmize the data:
sc.pp.log1p(adata)
# Identify highly-variable genes.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
X = X[:, adata.var.highly_variable]
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
# Clustering the neighborhood graph
sc.tl.leiden(adata, resolution=0.4, key_added="clusters")
adata.obs["clusters"] = ["0" if item == "4" else item for item in adata.obs['clusters']]
adata.obs["clusters"] = ["0" if item == "5" else item for item in adata.obs['clusters']]
adata.obs["clusters"] = ["0" if item == "3" else item for item in adata.obs['clusters']]
adata.obs["clusters"] = pd.Series(adata.obs["clusters"], name="clusters", dtype="category")
# Rename clusters
new_cluster_names = ['Basal-1', 'Basal-4', 'Basal-2/3']
adata.rename_categories('clusters', new_cluster_names)
# Find differentially expressed features
sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon',
                        corr_method='benjamini-hochberg', tie_correct=True)
# Filter markers
temp_dict = {}
for key, items in markers_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
markers_dict = temp_dict
markers = [f for k, item in markers_dict.items() for f in item]
basal_markers = ['ABI3BP', 'ATP1B1', 'BCAM', 'COL17A1', 'CXCL17', 'DCN', 'DST', 'DUT',
                 'ITGA6', 'ITGB4', 'KRT15', 'KRT5', 'LAMB3', 'NGFR', 'PDPN', 'TRP63', 'TSPAN1', 'WFDC2']
basal_markers = sorted(list(set(basal_markers)))
secretory_markers = ['AGR2', 'BPIFA1', 'BPIFB1', 'MSLN', 'MUC5B', 'SCGB3A1',
                     'SCGB3A2', 'WFDC2']
secretory_markers = sorted(list(set(secretory_markers)))
selected_features_dict = {'Basal': basal_markers,
                          'Secretory': secretory_markers}
# selected_features_dict = {'Basal': ['COL17A1', 'KRT15', 'KRT5', 'LAMB3', 'TRP63', 
#                                     'TSPAN1'],
#                           'Secretory': ['AGR2', 'BPIFA1', 'MSLN', 'MUC5B', 
#                                         'SCGB1A1', 'SCGB3A2']}
temp_dict = {}
for key, items in selected_features_dict.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
selected_features_dict = temp_dict
selected_features = [f for k, item in selected_features_dict.items() for f in item]
temp_dict = {}
for key, items in basal_subsets_markers.items():
    temp = []
    for item in items:
        if item.upper() in adata.var_names:
            temp.append(item.upper())
    if len(temp) > 0:
        temp_dict[key] = temp
basal_subsets_markers = temp_dict
# Plot UMAP
adata.uns["donors_colors"] = ["#F05454", "#59CE8F"]
adata.uns["clusters_colors"] = ['#4c72b0', '#c44e52', '#dd8452']
with plt.rc_context({'figure.figsize': (8, 6), 'axes.titlesize': '24'}):
    sc.pl.umap(adata, color=['clusters'] + ['donors'],
               use_raw=False, add_outline=False, legend_loc='on data',
               legend_fontsize=30, legend_fontoutline=0, frameon=False)
# Plot distributions of basal cells in injury vs uninjury conditions
basal_populations = list()
for d in sorted(set(adata.obs["donors"])):
    df = adata[adata.obs["donors"] == d].obs.groupby(["clusters"]).count()["donors"]
    df /= df.sum()
    temp = df.to_list()
    temp.insert(0, d)
    basal_populations.append(temp)
basal_populations = pd.DataFrame(basal_populations, columns=["Cluster"] + df.index.to_list())
# Use static colors
basal_palette = {"Basal-1": "#4c72b0", "Basal-2/3": "#dd8452",
                 "Basal-4": "#c44e52"}
plt.figure(figsize=(10, 8))
ax = basal_populations.plot(kind='bar', stacked=True, width=0.9, color=basal_palette)
ticks = [str(float(t.get_text()) * 100) for t in ax.get_yticklabels()]
ax.set_yticklabels(ticks, fontsize=16)
ax.set_xticklabels(["Injured", "Uninjured"], rotation=45, fontsize=20)
ax.set_ylabel("Percentage of basal cells", fontsize=20)
ax.set_xlabel(None)
ax.legend(title=None, fontsize=26, ncol=1, bbox_to_anchor=(1.005, 1),
          loc=2, borderaxespad=0., frameon=False)
# Violin plot
adata.X = X
adata.var_names_make_unique()
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.violin(adata, selected_features_dict["Basal"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
    sc.pl.violin(adata, selected_features_dict["Secretory"],
                 groupby='clusters', xlabel="Clusters", stripplot=False,
                 inner='box')
# Dotplot
sc.pp.scale(adata, max_value=10)
with plt.rc_context({'figure.figsize': (8, 10), 'figure.labelsize': '30',
                     'axes.titlesize': '30', 'axes.labelsize': '30',
                     'xtick.labelsize': '30', 'ytick.labelsize': '30'}):
    sc.pl.dotplot(adata, selected_features_dict, groupby='clusters',
                  title=None, colorbar_title="Mean expression values",
                  size_title="Fraction of expressed \n cells (%)")
# Heatmaps
adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X
with plt.rc_context({'figure.labelsize': '30', 'axes.titlesize': '20',
                     'axes.labelsize': '30', 'xtick.labelsize': '35',
                     'ytick.labelsize': '12'}):
    sc.pl.heatmap(adata, selected_features_dict, groupby='clusters',
                  layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', dendrogram=False,
                  swap_axes=False, figsize=(10, 2))
