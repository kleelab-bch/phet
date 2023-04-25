from copy import deepcopy

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import iqr
from scipy.stats import zscore


class SeuratHVF:
    def __init__(self, per_condition: bool = False, log_transform: bool = False,
                 num_top_features: int = None, min_disp: float = 0.5,
                 min_mean: float = 0.0125, max_mean: float = 3):
        self.per_condition = per_condition
        self.log_transform = log_transform
        self.num_top_features = num_top_features
        self.min_disp = min_disp
        self.min_mean = min_mean
        self.max_mean = max_mean

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        # Shift data 
        min_value = X.min(0)
        if len(np.where(min_value < 0)[0]) > 0:
            X = X - min_value + 1
        adata = ad.AnnData(X=X)
        # Logarithm transformation
        if self.log_transform:
            sc.pp.log1p(adata)
            np.nan_to_num(adata.X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.zeros((num_features))
        if self.per_condition:
            X = deepcopy(adata.X)
            for class_idx in range(num_classes):
                class_idx = np.where(y == class_idx)[0]
                # Identify highly-variable features.
                adata = ad.AnnData(X=X[class_idx])
                sc.pp.highly_variable_genes(adata, n_top_genes=self.num_top_features,
                                            min_disp=self.min_disp, min_mean=self.min_mean,
                                            max_mean=self.max_mean)
                features_idx = adata.var["highly_variable"]
                results[features_idx] += np.absolute(adata.var[features_idx]["dispersions_norm"])
        else:
            # Identify highly-variable features.
            sc.pp.highly_variable_genes(adata, n_top_genes=self.num_top_features,
                                        min_disp=self.min_disp, min_mean=self.min_mean,
                                        max_mean=self.max_mean)
            features_idx = adata.var["highly_variable"]
            results[features_idx] = np.absolute(adata.var[features_idx]["dispersions_norm"])
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results


class HIQR:
    def __init__(self, per_condition: bool = False, normalize: str = "zscore",
                 iqr_range: int = (25, 75), ):
        self.per_condition = per_condition
        self.normalize = normalize
        self.iqr_range = iqr_range

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        # Extract properties
        num_classes = len(np.unique(y))
        num_features = X.shape[1]

        if self.normalize == "robust":
            # Robustly estimate median by classes
            med = list()
            for i in range(num_classes):
                example_idx = np.where(y == i)[0]
                example_med = np.median(X[example_idx], axis=0)
                temp = np.absolute(X[example_idx] - example_med)
                med.append(temp)
            med = np.median(np.concatenate(med), axis=0)
            X = X / med
            del example_idx, example_med, temp, med
        elif self.normalize == "zscore":
            X = zscore(X, axis=0)

        results = np.zeros((num_features))
        if self.per_condition:
            for feature_idx in range(num_features):
                for class_idx in range(num_classes):
                    class_idx = np.where(y == class_idx)[0]
                    results[feature_idx] += iqr(X[class_idx, feature_idx], rng=self.iqr_range, scale=1.0)
        else:
            for feature_idx in range(num_features):
                results[feature_idx] = iqr(X[:, feature_idx], rng=self.iqr_range, scale=1.0)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results
