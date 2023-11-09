from itertools import combinations
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import iqr, zscore, ttest_ind
from scipy.stats import ks_2samp
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.weightstats import ztest

SEED_VALUE = 0.001


class PHeT:
    def __init__(self, normalize: Optional[str] = "zscore", iqr_range: Optional[tuple] = (25, 75),
                 num_subsamples: int = 1000, subsampling_size: int = None, disp_type: str = "iqr",
                 feature_weight: list = None, num_jobs: int = 2):
        self.normalize = normalize  # robust, zscore, or log
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.delta_type = disp_type
        if disp_type == "hvf":
            if self.normalize is not None:
                self.normalize = "log"
        if len(feature_weight) < 2:
            feature_weight = [0.4, 0.3, 0.2, 0.1]
        self.feature_weight = np.array(feature_weight) / np.sum(feature_weight)
        self.num_jobs = num_jobs

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        # Extract properties
        num_examples, num_features = X.shape
        # Check if classes information is not provided (unsupervised analysis)
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if control_class == case_class:
            temp = "Please provide two distinct groups ids!"
            raise Exception(temp)
        if control_class not in np.unique(y) or case_class not in np.unique(y):
            temp = "Please provide valid control/case group ids!"
            raise Exception(temp)
        num_classes = len(np.unique(y))

        # Total number of combinations
        num_combinations = len(list(combinations(range(num_classes), 2)))

        if self.delta_type == "hvf":
            # Shift data 
            min_value = X.min(0)
            if len(np.where(min_value < 0)[0]) > 0:
                X = X - min_value + 1
            # Logarithm transformation
            if self.normalize == "log":
                X = np.log(X + 1)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if self.normalize == "robust":
                # Robustly estimate median by classes
                med = list()
                for class_idx in range(num_classes):
                    if num_classes > 1:
                        class_idx = np.where(y == class_idx)[0]
                    else:
                        class_idx = range(num_examples)
                    example_med = np.median(X[class_idx], axis=0)
                    temp = np.absolute(X[class_idx] - example_med)
                    med.append(temp)
                med = np.median(np.concatenate(med), axis=0)
                X = X / med
                del class_idx, example_med, temp, med
            elif self.normalize == "zscore":
                X = zscore(X, axis=0)
            elif self.normalize == "log":
                min_value = X.min(0)
                if len(np.where(min_value < 0)[0]) > 0:
                    X = X - min_value + 1
                X = np.log1p(X)
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Define the subsampling size for iterative subsampling
        num_subsamples = self.num_subsamples
        subsampling_size = self.subsampling_size
        if subsampling_size is None:
            temp = list()
            for class_idx in range(num_classes):
                examples_idx = np.where(y == class_idx)[0]
                temp.append(len(examples_idx))
            if np.min(temp) <= 2:
                temp = int(np.min(temp))
            else:
                temp = int(np.sqrt(np.min(temp)))
            subsampling_size = temp
        else:
            temp = list()
            for class_idx in range(num_classes):
                examples_idx = np.where(y == class_idx)[0]
                temp.append(len(examples_idx))
            temp = np.min(temp)
            if subsampling_size > temp:
                subsampling_size = temp

        # Step 1: Iterative subsampling process to select and rank significant features
        P = np.zeros((num_features, num_subsamples))
        r = np.zeros((num_features,))
        combination_idx = 0
        temp = np.zeros((num_features, num_combinations))
        for i, j in combinations(range(num_classes), 2):
            for sample_idx in range(num_subsamples):
                examples_i = np.where(y == i)[0]
                examples_j = np.where(y == j)[0]
                examples_i = np.random.choice(a=examples_i, size=subsampling_size, replace=False)
                examples_j = np.random.choice(a=examples_j, size=subsampling_size, replace=False)
                if self.delta_type == "hvf":
                    adata = ad.AnnData(X=X[examples_i])
                    sc.pp.highly_variable_genes(adata, n_top_genes=num_features)
                    disp1 = adata.var["dispersions_norm"].to_numpy()
                    adata = ad.AnnData(X=X[examples_j])
                    sc.pp.highly_variable_genes(adata, n_top_genes=num_features)
                    disp2 = adata.var["dispersions_norm"].to_numpy()
                    delta_h = np.absolute(disp1 - disp2)
                    del adata
                else:
                    iq_range_i = iqr(X[examples_i], axis=0, rng=self.iqr_range, scale=1.0)
                    iq_range_j = iqr(X[examples_j], axis=0, rng=self.iqr_range, scale=1.0)
                    delta_h = np.absolute(iq_range_i - iq_range_j)
                np.nan_to_num(delta_h, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                temp[:, combination_idx] += delta_h / num_subsamples
                if subsampling_size < 30:
                    _, pvalue = ttest_ind(X[examples_i], X[examples_j])
                else:
                    _, pvalue = ztest(X[examples_i], X[examples_j])
                P[:, sample_idx] += pvalue / num_combinations
            combination_idx += 1
        r = np.max(temp, axis=1)
        if self.delta_type == "hvf":
            del temp, disp1, disp2, delta_h
        else:
            del temp, iq_range_i, iq_range_j, delta_h
        np.nan_to_num(r, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 2: Apply Fisher's method for combined probability
        f = -2 * np.log(P)
        f[f == np.inf] = 0
        f = np.sum(f, axis=1)
        # Standardize I to be used in the final ranking of features. This is useful
        # to calculate p-values)
        f = (f - 2 * num_subsamples) / np.sqrt(4 * num_subsamples)
        # Keep only the highest Fisher's statistics
        f[f < 0] = SEED_VALUE
        f = np.absolute(f)
        np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        del P

        # Step 3: Discriminative power calculation
        temp = []
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp.append(len(examples_idx))
        min_size = np.min(temp)
        slice_size = subsampling_size
        o = np.zeros((num_features, 1))
        for feature_idx in range(num_features):
            temp_pvalues = list()
            for i, j in combinations(range(num_classes), 2):
                temp = []
                examples_i = np.where(y == i)[0]
                examples_j = np.where(y == j)[0]
                examples_i = X[examples_i, feature_idx]
                examples_j = X[examples_j, feature_idx]
                examples_i = np.random.permutation(examples_i)
                examples_j = np.random.permutation(examples_j)
                for slice_idx in np.arange(0, min_size, slice_size):
                    temp_size = slice_size
                    if slice_idx + slice_size >= min_size:
                        temp_size = np.min((examples_j[slice_idx:].shape[0],
                                            examples_i[slice_idx:].shape[0]))
                    pvalue = ks_2samp(examples_i[slice_idx: slice_idx + temp_size],
                                      examples_j[slice_idx: slice_idx + temp_size])[1]
                    temp.append(pvalue)
                pvalue = np.min(temp)
                temp_pvalues.append(pvalue)
            o[feature_idx] = np.mean(temp_pvalues)

        temp = KBinsDiscretizer(n_bins=len(self.feature_weight), encode="ordinal",
                                strategy="uniform").fit_transform(o)
        o = np.zeros((num_features, len(self.feature_weight)), dtype=np.int8)
        for bin_idx in range(len(self.feature_weight)):
            o[np.where(temp == bin_idx)[0], bin_idx] = 1
        del temp

        # Step 4: Estimating features statistics based on combined parameters (r, o, f)
        r /= r.sum()
        o = self.feature_weight.dot(o.T)
        f = np.multiply(f, o)
        f /= f.sum()
        results = r + f

        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))

        return results
