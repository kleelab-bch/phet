import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

EPSILON = np.finfo(np.float).eps

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
sns.set_theme()

plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=16)


def color_map(num_colors, cmap=plt.cm.cool, force_colors: bool = True):
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if force_colors:
        # force colors entry to be red, green, blue, and others
        cmaplist[0] = (1.0, 0.0, 0.0, 1.0)
        cmaplist[1] = (0.0, 0.5, 0.0, 1.0)
        cmaplist[2] = (0.0, 0.0, 1.0, 1.0)
        cmaplist[3] = (0.5, 0.5, 0.5, 1.0)
        cmaplist[4] = (0.0, 1.0, 0.0, 1.0)
        cmaplist[5] = (0.0, 0.5, 0.5, 1.0)
        cmaplist[6] = (0.5, 0.0, 0.0, 1.0)
        cmaplist[7] = (1.0, 1.0, 0.0, 1.0)
        cmaplist[8] = (0.0, 1.0, 1.0, 1.0)
        cmaplist[9] = (1.0, 0.0, 1.0, 1.0)
        cmaplist[10] = (0.0, 0.0, 0.0, 0.9)
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    color_labels = [cmap(i) for i in range(cmap.N)]
    color_labels = color_labels[:num_colors]
    return color_labels


def plot_silhouette(X, X_reducer, num_clusters, cluster_labels, cluster_labels_colors: list = None,
                    unique_cluster_colors: list = None, scatter_size: int = 30, save_name: str = "",
                    save_path: str = ""):
    cluster_centers = []
    for idx in range(num_clusters):
        temp = np.nonzero(cluster_labels == idx)[0]
        temp = X_reducer[temp]
        cluster_centers.append(np.mean(temp, axis=0))
    cluster_centers = np.array(cluster_centers)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (num_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    if unique_cluster_colors is None:
        unique_cluster_colors = [cm.nipy_spectral(float(idx) / num_clusters) for idx in range(num_clusters)]

    for idx in range(num_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == idx]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=unique_cluster_colors[idx],
                          edgecolor=unique_cluster_colors[idx], alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(idx))
        y_lower = y_upper + 10

    # first subplot
    ax1.set_title("Silhouette plot for various clusters")
    ax1.set_xlabel("Silhouette coefficient values (average %.4f)" % silhouette_avg)
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # second subplot
    ax2.scatter(X_reducer[:, 0], X_reducer[:, 1], marker=".", s=scatter_size, lw=0, alpha=0.7, c=cluster_labels_colors,
                edgecolor="k")
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="o", c="white", alpha=1, s=200,
                edgecolor="k")
    for idx, c in enumerate(cluster_centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
    ax2.set_title("Visualization of the clustered data")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    plt.suptitle("Silhouette analysis for %d clusters" % num_clusters, fontsize=14, fontweight="bold")
    tmp = os.path.join(save_path, save_name + ".png")
    plt.savefig(tmp)
    plt.clf()
    plt.cla()
    plt.close(fig="all")
    return silhouette_avg


def plot_umap(X, color_gradient, cluster_labels=None, marker_size: int = 30, fig_title: str = "Title",
              ylabel: str = "Feature", cbar: bool = True, file_name: str = "temp", save_path: str = "."):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    plt.scatter(X[:, 0], X[:, 1], alpha=1, c=color_gradient, cmap='hot', s=marker_size)
    if cbar:
        cbar = plt.colorbar(ax=ax)
        cbar.ax.set_ylabel(ylabel, labelpad=10, rotation=270, fontsize=16, fontweight="bold")
    if cluster_labels is not None:
        cluster_centers = []
        for idx in np.unique(cluster_labels):
            temp = np.nonzero(cluster_labels == idx)[0]
            temp = X[temp]
            cluster_centers.append(np.mean(temp, axis=0))
        cluster_centers = np.array(cluster_centers)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="o", c="white", alpha=1, s=200,
                    edgecolor="k")
        for idx, c in enumerate(cluster_centers):
            plt.scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.suptitle(fig_title, fontsize=16, fontweight="bold")

    file_path = os.path.join(save_path, file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.cla()
    plt.close(fig="all")


def plot_boxplot(X, attribute_names: list, cluster_colors: list, cluster_labels: list,
                 segment_interval: int = 20, scale_features: bool = True, tag: str = "coefficients",
                 save_path: str = ".", verbose: bool = True):
    num_examples, time_frame, num_attributes = X.shape

    # Scale data
    if scale_features:
        scaler = StandardScaler()
        X = X.reshape(num_examples * time_frame, num_attributes)
        X = scaler.fit_transform(X)
        X = X.reshape(num_examples, time_frame, num_attributes)

    # Iterate over all attributes
    current_progress = 0
    total_progress = num_attributes
    for attribute_idx, attribute_name in enumerate(attribute_names):
        current_progress += 1
        desc = '\t\t--> Progress: {0:.2f}%...'.format((current_progress / total_progress) * 100)
        if verbose:
            if current_progress == total_progress:
                print(desc)
            else:
                print(desc, end="\r")

        X_attribute = X[:, :, attribute_idx]

        # Plot by segments
        figs_steps = plt.figure(figsize=(20, 22))
        axs_steps = list()
        range_time = list(range(0, time_frame, segment_interval)) + [-1]
        for idx, start_time in enumerate(range_time):
            if start_time == -1:
                start_time = 0
                end_time = time_frame
            else:
                if len(range_time) == idx + 2:
                    end_time = time_frame
                else:
                    end_time = range_time[idx + 1]
            X_segment = X_attribute[:, start_time: end_time]
            X_segment = [np.mean(X_segment[cluster_labels == cluster_idx], axis=1) for cluster_idx in
                         np.unique(cluster_labels)]
            # plot boxplot
            axs_steps.append(figs_steps.add_subplot(int(np.ceil(len(range_time) / 2)), 2, idx + 1))
            bplot = axs_steps[idx].boxplot(X_segment, labels=np.unique(cluster_labels),
                                           medianprops=dict(linestyle='-', linewidth=2, color='black'),
                                           flierprops=dict(marker='o', markerfacecolor='black', markersize=18,
                                                           linestyle='none'),
                                           vert=True, patch_artist=True,
                                           sym='+', whis=1.5)

            # Add a horizontal grid to the plot, but make it very light in color
            # so we can use it for reading data values but not be distracting
            axs_steps[idx].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

            # Adjust ticks and labels
            axs_steps[idx].set_title('Time %d-%d' % (start_time, end_time), fontsize=18, fontweight="bold")
            axs_steps[idx].set_xticklabels(labels=np.unique(cluster_labels), fontsize=20, fontweight="bold")
            axs_steps[idx].set_ylabel('Standard values', fontsize=16)
            max_val = max([item.max() for item in X_segment]) + 0.5
            min_val = min([item.min() for item in X_segment]) - 0.5
            axs_steps[idx].set_ylim(min_val, max_val)

            # Now fill the boxes with desired colors
            for i in np.unique(cluster_labels):
                bplot['boxes'][i].set_facecolor(cluster_colors[i])

        figs_steps.suptitle(' '.join(attribute_name.split("_")).capitalize(), fontsize=26,
                            fontweight="bold")
        figs_steps.tight_layout()
        temp = "boxplot_%s_%s.png" % (attribute_name, tag)
        file_path = os.path.join(save_path, temp)
        figs_steps.savefig(file_path)
        del figs_steps, axs_steps
        plt.clf()
        plt.cla()
        plt.close(fig="all")
