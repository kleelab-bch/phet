import logging
import os
import warnings

logging.getLogger('tensorflow').disabled = True
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from itertools import combinations
from typing import Optional, Literal

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import f_oneway, kruskal, false_discovery_control
from scipy.stats import iqr, zscore, ttest_ind
from scipy.stats import ks_2samp, norm
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.weightstats import ztest

SEED_VALUE = 0.001


class PHet:
    def __init__(self, normalize: Literal["robust", "zscore", "log"] | None = "zscore",
                 iqr_range: Optional[tuple] = (25, 75), num_subsamples: int = 1000,
                 subsampling_size: Literal["optimum", "sqrt"] | int = "sqrt",
                 delta_type: Literal["iqr", "hvf"] = "iqr",
                 group_test: Literal["anova", "kruskal"] = "anova",
                 multiconditions_strategy: Literal["pairwise", "one_vs_all"] = "one_vs_all",
                 multiconditions_aggregation: Literal["mean", "max"] = "max",
                 adjust_pvalue: bool = False, feature_weight: list = None):
        self.normalize = normalize  # robust, zscore, or log
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.delta_type = delta_type
        self.group_test = group_test
        self.multiconditions_strategy = multiconditions_strategy
        self.multiconditions_aggregation = multiconditions_aggregation
        self.adjust_pvalue = adjust_pvalue
        if delta_type == "hvf":
            if self.normalize is not None:
                self.normalize = "log"
        if len(feature_weight) < 2:
            feature_weight = [0.4, 0.3, 0.2, 0.1]
        self.feature_weight = np.array(feature_weight) / np.sum(feature_weight)

    def __optimum_sample_size(self, control_size: int, case_size: int, alpha: float = 0.05,
                              margin_error: float = 0.1):
        total_samples = control_size + case_size
        p = control_size / total_samples
        q = 1 - p
        z = np.sign(norm.ppf(alpha)) * norm.ppf(alpha)
        numerator = (z ** 2 * p * (1 - q)) / margin_error ** 2
        denominator = 1 + (numerator / total_samples)
        sample_size = int(np.ceil(numerator / denominator))
        return sample_size

    def fit_predict(self, X, y, sample_ids: list | None = None, subtypes: list | None = None):
        # Extract properties
        num_examples, num_features = X.shape
        # Check if classes information is not provided
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        if num_classes < 2:
            temp = "More than two valid groups are allowed!"
            raise Exception(temp)
        multiple_conditions = False
        if num_classes > 2:
            multiple_conditions = True
        pseudobulk_de = False
        if sample_ids:
            if len(sample_ids) != num_examples:
                temp = "The number of sample IDs provided does not match the number of example!"
                raise Exception(temp)
            if subtypes:
                if len(subtypes) != num_examples:
                    temp = "The number of subtypes provided does not match the number of examples!"
                    raise Exception(temp)
            pseudobulk_de = True

        if pseudobulk_de:
            if subtypes:
                temp = pd.DataFrame([sample_ids, [str(idx) for idx in y], subtypes]).T
                temp_columns = ["batch", "condition", "cell_type"]
            else:
                temp = pd.DataFrame([sample_ids, [str(idx) for idx in y]]).T
                temp_columns = ["batch", "condition"]
            pdata = sc.AnnData(X=X, obs=temp)
            pdata.obs.columns = temp_columns
            pdata = dc.get_pseudobulk(pdata, sample_col="batch", groups_col=temp_columns[1:], mode="sum",
                                      min_cells=0, min_counts=0)
            # Normalize and scale
            sc.pp.normalize_total(pdata, target_sum=1e4)
            sc.pp.log1p(pdata)
            sc.pp.scale(pdata, max_value=10)
            X_pseudobulk = pdata.X
            y_pseudobulk = pdata.obs["condition"].astype(int).to_list()
            del pdata

        if self.delta_type == "hvf":
            # Shift data
            min_value = X.min(0)
            if len(np.where(min_value < 0)[0]) > 0:
                X = X - min_value + 1
                # Logarithm transformation
            if self.normalize == "log":
                X = np.log(X + 1)
        else:
            if self.normalize == "robust":
                # Robustly estimate median by classes
                med = list()
                for class_idx in unique_classes:
                    if num_classes > 1:
                        class_idx = np.where(y == class_idx)[0]
                    else:
                        class_idx = range(num_examples)
                    example_med = np.median(X[class_idx], axis=0)
                    temp = np.absolute(X[class_idx] - example_med)
                    med.append(temp)
                med = np.median(np.concatenate(med), axis=0)
                X = X / med
                if pseudobulk_de:
                    med = list()
                    for class_idx in unique_classes:
                        if num_classes > 1:
                            class_idx = np.where(y_pseudobulk == class_idx)[0]
                        else:
                            class_idx = range(X_pseudobulk.shape[0])
                        example_med = np.median(X_pseudobulk[class_idx], axis=0)
                        temp = np.absolute(X_pseudobulk[class_idx] - example_med)
                        med.append(temp)
                    med = np.median(np.concatenate(med), axis=0)
                    X_pseudobulk = X_pseudobulk / med
                del class_idx, example_med, temp, med
            elif self.normalize == "zscore":
                X = zscore(X, axis=0)
                if pseudobulk_de:
                    X_pseudobulk = zscore(X_pseudobulk, axis=0)
            elif self.normalize == "log":
                min_value = X.min(0)
                if len(np.where(min_value < 0)[0]) > 0:
                    X = X - min_value + 1
                X = np.log1p(X)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Total number of combinations
        num_combinations = len(list(combinations(range(num_classes), 2)))
        total_combinations = list(combinations(unique_classes, 2))
        if multiple_conditions and self.multiconditions_strategy == "one_vs_all":
            num_combinations = num_classes
            total_combinations = list(zip(range(num_classes), [0] * num_classes))

        # Define the subsampling size for iterative subsampling
        subsampling_size = self.subsampling_size
        if subsampling_size == "optimum":
            subsampling_size = []
            for i, j in total_combinations:
                control_size = len(np.where(y == i)[0])
                if multiple_conditions and self.multiconditions_strategy == "one_vs_all":
                    case_size = len(np.where(y != i)[0])
                else:
                    case_size = len(np.where(y == j)[0])
                temp = self.__optimum_sample_size(control_size=control_size,
                                                  case_size=case_size, alpha=0.05,
                                                  margin_error=0.1)
                subsampling_size.append(temp)
            subsampling_size = int(np.min(subsampling_size))
        elif subsampling_size == "sqrt":
            temp = list()
            for class_idx in unique_classes:
                examples_idx = np.where(y == class_idx)[0]
                temp.append(len(examples_idx))
            if np.min(temp) <= 2 or int(np.sqrt(np.min(temp))) < 8:
                temp = int(np.min(temp))
            else:
                temp = int(np.sqrt(np.min(temp)))
            subsampling_size = temp

        # Step 1: Iterative subsampling process to select and rank significant features
        num_subsamples = self.num_subsamples
        if pseudobulk_de:
            num_subsamples = 1
        P = np.zeros((num_features, num_subsamples))
        # multiple conditions
        if multiple_conditions:
            examples_idx = []
            replace = False
            for class_idx in unique_classes:
                if pseudobulk_de:
                    examples = X_pseudobulk[y_pseudobulk == class_idx]
                    examples_idx.append(examples)
                else:
                    examples = X[y == class_idx]
                    examples_idx.append(examples)
                    if subsampling_size > examples.shape[0]:
                        replace = True
            for sample_idx in range(num_subsamples):
                subsamples = []
                for examples in examples_idx:
                    if pseudobulk_de:
                        temp_idx = list(range(examples.shape[0]))
                    else:
                        temp_idx = np.random.choice(a=range(examples.shape[0]), size=subsampling_size,
                                                    replace=replace)
                    subsamples.append(examples[temp_idx])
                if self.group_test == "anova":
                    _, pvalue = f_oneway(*subsamples)
                else:
                    if subsampling_size > 5:
                        _, pvalue = kruskal(*subsamples)
                    else:
                        _, pvalue = f_oneway(*subsamples)
                np.nan_to_num(pvalue, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                if self.adjust_pvalue:
                    pvalue = false_discovery_control(pvalue, method="bh")
                P[:, sample_idx] = pvalue

        combination_idx = 0
        num_subsamples = self.num_subsamples
        r = np.zeros((num_features, num_subsamples, num_combinations))
        for i, j in total_combinations:
            examples_i = np.where(y == i)[0]
            if multiple_conditions and self.multiconditions_strategy == "one_vs_all":
                examples_j = np.where(y != i)[0]
            else:
                examples_j = np.where(y == j)[0]
            replace = False
            # Replacement is enabled when example size from any conditions is below the subsampling size
            if subsampling_size > len(examples_i) or subsampling_size > len(examples_j):
                replace = True
            for sample_idx in range(num_subsamples):
                subsample_i = np.random.choice(a=examples_i, size=subsampling_size, replace=replace)
                subsample_j = np.random.choice(a=examples_j, size=subsampling_size, replace=replace)
                if self.delta_type == "hvf":
                    adata = ad.AnnData(X=X[subsample_i])
                    sc.pp.highly_variable_genes(adata, n_top_genes=num_features)
                    disp1 = adata.var["dispersions_norm"].to_numpy()
                    adata = ad.AnnData(X=X[subsample_j])
                    sc.pp.highly_variable_genes(adata, n_top_genes=num_features)
                    disp2 = adata.var["dispersions_norm"].to_numpy()
                    delta_h = np.absolute(disp1 - disp2)
                    del adata
                else:
                    iq_range_i = iqr(X[subsample_i], axis=0, rng=self.iqr_range, scale=1.0)
                    iq_range_j = iqr(X[subsample_j], axis=0, rng=self.iqr_range, scale=1.0)
                    delta_h = np.absolute(iq_range_i - iq_range_j)
                np.nan_to_num(delta_h, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                r[:, sample_idx, combination_idx] = delta_h
                if not multiple_conditions:
                    if pseudobulk_de:
                        sample_idx = 0
                        subsample_i = np.where(y_pseudobulk == i)[0]
                        subsample_j = np.where(y_pseudobulk == j)[0]
                        _, pvalue = ttest_ind(X_pseudobulk[subsample_i], X_pseudobulk[subsample_j])
                    else:
                        if subsampling_size < 30:
                            _, pvalue = ttest_ind(X[subsample_i], X[subsample_j])
                        else:
                            _, pvalue = ztest(X[subsample_i], X[subsample_j])
                    np.nan_to_num(pvalue, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    if self.adjust_pvalue:
                        pvalue = false_discovery_control(pvalue, method="bh")
                    P[:, sample_idx] = pvalue
            combination_idx += 1
        r = np.mean(r, axis=1)
        if self.multiconditions_aggregation == "mean":
            r = np.mean(r, axis=1)
        else:
            r = np.max(r, axis=1)
        if self.delta_type == "hvf":
            del disp1, disp2, delta_h
        else:
            del iq_range_i, iq_range_j, delta_h
        np.nan_to_num(r, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 2: Apply Fisher's method for combined probability
        f = -2 * np.log(P)
        f[f == np.inf] = 0
        f = np.sum(f, axis=1)
        # f has a chi-squared distribution with 2 * num_subsamples degrees of freedom.
        # So, we can standardize f to be used in the final ranking of features.
        # Mean of chi-squared is 2 * num_subsamples while the standard deviation
        # is np.sqrt(4 * num_subsamples).
        if pseudobulk_de:
            num_subsamples = 1
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
            for i, j in total_combinations:
                temp = []
                examples_i = np.where(y == i)[0]
                if multiple_conditions and self.multiconditions_strategy == "one_vs_all":
                    examples_j = np.where(y != i)[0]
                else:
                    examples_j = np.where(y == j)[0]
                subsample_i = X[examples_i, feature_idx]
                subsample_j = X[examples_j, feature_idx]
                subsample_i = np.random.permutation(subsample_i)
                subsample_j = np.random.permutation(subsample_j)
                for slice_idx in np.arange(0, min_size, slice_size):
                    temp_size = slice_size
                    if slice_idx + slice_size >= min_size:
                        temp_size = np.min((subsample_j[slice_idx:].shape[0],
                                            subsample_i[slice_idx:].shape[0]))
                    pvalue = ks_2samp(subsample_i[slice_idx: slice_idx + temp_size],
                                      subsample_j[slice_idx: slice_idx + temp_size])[1]
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
