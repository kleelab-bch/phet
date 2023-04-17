from copy import deepcopy

import anndata as ad
import numpy as np
import scanpy as sc


class DeltaHVFMean:
    def __init__(self, calculate_deltamean: bool = True, num_top_features: int = None, min_disp: float = 0.5,
                 min_mean: float = 0.0125, max_mean: float = 3):
        self.calculate_deltamean = calculate_deltamean
        self.num_top_features = num_top_features
        self.min_disp = min_disp
        self.min_mean = min_mean
        self.max_mean = max_mean

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):

        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        results = np.zeros((num_features))

        adata = ad.AnnData(X=X)
        # Total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell,
        # so that counts become comparable among cells.
        sc.pp.normalize_total(adata, target_sum=1e4)
        # Logarithmize the data:
        sc.pp.log1p(adata)
        np.nan_to_num(adata.X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        delta_disps = list()
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
                delta_disps.append(np.abs(disp1 - disp2))
                delta_means.append(np.abs(mean1 - mean2))
        results = np.max(delta_disps, axis=0)
        if self.calculate_deltamean:
            results += np.max(delta_means, axis=0)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results
