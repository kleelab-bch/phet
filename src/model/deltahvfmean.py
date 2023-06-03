from copy import deepcopy

import anndata as ad
import numpy as np
import scanpy as sc


class DeltaHVFMean:
    def __init__(self, calculate_deltamean: bool = True, log_transform: bool = False,
                 num_top_features: int = None, min_disp: float = 0.5,
                 min_mean: float = 0.0125, max_mean: float = 3):
        self.calculate_deltamean = calculate_deltamean
        self.log_transform = log_transform
        self.num_top_features = num_top_features
        self.min_disp = min_disp
        self.min_mean = min_mean
        self.max_mean = max_mean

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        results = np.zeros((num_features))
        # Shift data 
        min_value = X.min(0)
        if len(np.where(min_value < 0)[0]) > 0:
            X = X - min_value + 1
        adata = ad.AnnData(X=X)
        # Logarithm transformation
        if self.log_transform:
            sc.pp.log1p(adata)
            np.nan_to_num(adata.X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        delta_vars = list()
        delta_means = list()
        X = deepcopy(adata.X)
        for i in range(num_classes):
            examples_i = np.where(y == i)[0]
            adata = ad.AnnData(X=X[examples_i])
            sc.pp.highly_variable_genes(adata, n_top_genes=self.num_top_features,
                                        min_disp=self.min_disp, min_mean=self.min_mean,
                                        max_mean=self.max_mean)
            features_idx = adata.var["highly_variable"]
            disp1 = adata.var[features_idx]["dispersions_norm"].to_numpy()
            mean1 = np.mean(X[examples_i], axis=0)
            for j in range(i + 1, num_classes):
                examples_j = np.where(y == j)[0]
                adata = ad.AnnData(X=X[examples_j])
                sc.pp.highly_variable_genes(adata, n_top_genes=self.num_top_features,
                                            min_disp=self.min_disp, min_mean=self.min_mean,
                                            max_mean=self.max_mean)
                features_idx = adata.var["highly_variable"]
                disp2 = adata.var[features_idx]["dispersions_norm"].to_numpy()
                mean2 = np.mean(X[examples_j], axis=0)
                delta_vars.append(np.abs(disp1 - disp2))
                delta_means.append(np.abs(mean1 - mean2))
        results = np.max(delta_vars, axis=0)
        if self.calculate_deltamean:
            results += np.max(delta_means, axis=0)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results
