import os
import shutil
import warnings

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.cluster import SpectralCoclustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

EPSILON = np.finfo(np.float).eps

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
sns.set_theme()

plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=16)


def create_directory(name_folder: str = "test", save_path: str = "."):
    file_path = os.path.join(save_path, name_folder)
    if os.path.exists(save_path):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        os.makedirs(file_path)
        return file_path
    else:
        raise Exception("Invalid file path: {0}".format(save_path))


def normalize_laplacian(A, return_laplace: bool = False):
    D = A.sum(axis=1)
    D = np.diag(D)
    L = D - A
    with np.errstate(divide='ignore'):
        D = 1.0 / np.sqrt(D)
    D[np.isinf(D)] = 0
    D /= 2
    L = D.dot(L.dot(D))
    if return_laplace:
        return L
    A = D - L
    A[A <= 0] = 0
    return A


def features_selection(X, attribute_columns, variance_threshold, verbose: bool = False):
    if verbose:
        print("\t    >> Selecting features...")
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.feature_names_in_ = attribute_columns
    X = selector.fit_transform(X)
    attribute_columns = [attribute_columns[idx]
                         for idx, feat in enumerate(selector.get_support()) if feat]
    return X, attribute_columns


def dimensionality_reduction(X, num_neighbors: int = 5, num_components: int = 2, min_dist: float = 0.1,
                             perplexity: int = 30, reduction_method: str = "umap", num_epochs: int = 2000,
                             num_jobs: int = 2):
    num_examples, num_attributes = X.shape
    if reduction_method == "umap":
        init = "spectral"
        if num_attributes >= num_examples:
            init = "random"
        reducer = umap.UMAP(n_neighbors=num_neighbors, n_components=num_components, n_epochs=num_epochs, init=init,
                            min_dist=min_dist, n_jobs=num_jobs)
    elif reduction_method == "tsne":
        init = "pca"
        if num_attributes >= num_examples:
            init = "random"
        reducer = TSNE(n_components=num_components, perplexity=num_examples / 100, early_exaggeration=4,
                       learning_rate="auto", n_iter=num_epochs, init=init, random_state=12345, n_jobs=num_jobs)
    else:
        reducer = PCA(n_components=num_components, random_state=12345)
    X_reducer = reducer.fit_transform(X)
    return X_reducer


def clustering(X, cluster_type: str = "spectral", affinity: str = "nearest_neighbors", num_neighbors: int = 5,
               num_clusters: int = 4, num_jobs: int = 2, predict: bool = True):
    num_examples, num_attributes = X.shape
    if num_examples < num_clusters:
        num_clusters = num_examples
    if cluster_type == "kmeans":
        cls = MiniBatchKMeans(n_clusters=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "gmm":
        cls = GaussianMixture(n_components=num_clusters, max_iter=500, random_state=12345)
    elif cluster_type == "hdbscan":
        cls = hdbscan.HDBSCAN(min_samples=num_neighbors, min_cluster_size=num_clusters)
    elif cluster_type == "spectral":
        if num_neighbors > num_examples:
            num_neighbors = num_examples
        cls = SpectralClustering(n_clusters=num_clusters, eigen_solver="arpack", n_neighbors=num_neighbors,
                                 affinity=affinity, n_init=100, assign_labels='discretize', n_jobs=num_jobs,
                                 random_state=12345)
    elif cluster_type == "cocluster":
        cls = SpectralCoclustering(n_clusters=num_clusters, svd_method="arpack", random_state=12345)
    elif cluster_type == "agglomerative":
        cls = AgglomerativeClustering(n_clusters=num_clusters, affinity='manhattan', linkage='single')
    elif cluster_type == "affinity":
        cls = AffinityPropagation(affinity='euclidean', random_state=12345)
    if predict:
        cls = cls.fit_predict(X)
    else:
        cls.fit(X)
    return cls
