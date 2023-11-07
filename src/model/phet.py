'''
DeteCtion of celluLar hEterogeneity by anAlyzing variatioNs of cElls.
'''

from itertools import combinations
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import f_oneway, ks_2samp
from scipy.stats import iqr, norm, zscore, ttest_ind
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.weightstats import ztest
from utility.utils import clustering

SEED_VALUE = 0.001


class PHeT:
    def __init__(self, normalize: Optional[str] = "zscore", iqr_range: Optional[tuple] = (25, 75),
                 num_subsamples: int = 1000, subsampling_size: int = None, partition_by_anova: bool = False,
                 disp_type: str = "iqr", calculate_deltadisp: bool = True, calculate_deltamean: bool = False,
                 calculate_fisher: bool = True, calculate_disc_power: bool = True, binary_clustering: bool = True,
                 bin_pvalues: bool = True, feature_weight: list = None, weight_range: list = None,
                 num_jobs: int = 2):
        self.normalize = normalize  # robust, zscore, or log
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.partition_by_anova = partition_by_anova
        self.delta_type = disp_type
        if disp_type == "hvf":
            if self.normalize is not None:
                self.normalize = "log"
        self.calculate_deltadisp = calculate_deltadisp
        self.calculate_deltamean = calculate_deltamean
        self.calculate_fisher = calculate_fisher
        self.calculate_disc_power = calculate_disc_power
        self.binary_clustering = binary_clustering
        self.bin_pvalues = bin_pvalues
        if len(feature_weight) < 2:
            feature_weight = [0.4, 0.3, 0.2, 0.1]
        if not bin_pvalues:
            if len(feature_weight) > 4:
                feature_weight = [0.4, 0.3, 0.2, 0.1]
            if len(weight_range) != 3:
                weight_range = [0.1, 0.3, 0.5]
        self.feature_weight = np.array(feature_weight) / np.sum(feature_weight)
        self.weight_range = weight_range  # [0.1, 0.4, 0.8]
        self.num_jobs = num_jobs

    def __binary_partitioning(self, X):
        num_examples, num_features = X.shape
        M = np.zeros((num_features,))
        K = np.zeros((num_features, num_examples), dtype=np.int8)
        for feature_idx in range(num_features):
            temp = np.zeros((num_examples - 2))
            temp_X = np.sort(X[:, feature_idx])[::-1]
            order_list = np.argsort(X[:, feature_idx])[::-1]
            for k in range(1, num_examples - 1):
                S1 = temp_X[:k]
                S2 = temp_X[k:]
                if self.partition_by_anova:
                    _, pvalue = f_oneway(S1, S2)
                    temp[k - 1] = pvalue
                else:
                    # For the two subsets, the mean and sum of squares for each
                    # feature are calculated
                    mean1 = np.mean(S1)
                    mean2 = np.mean(S2)
                    SS1 = np.sum((S1 - mean1) ** 2)
                    SS2 = np.sum((S2 - mean2) ** 2)
                    temp[k - 1] = SS1 + SS2
            k = np.argmin(temp) + 1
            M[feature_idx] = np.min(temp)
            K[feature_idx, order_list[k:]] = 1
        k = np.argmin(M)
        y = K[k]
        if self.binary_clustering:
            y = clustering(X=K.T, cluster_type="agglomerative", affinity="euclidean", num_neighbors=5,
                           num_clusters=2, num_jobs=self.num_jobs, predict=True)
        return y

    def fit_predict(self, X, y=None, partition_data: bool = False, control_class: int = 0,
                    case_class: int = 1):
        # Extract properties
        num_examples, num_features = X.shape
        # Check if classes information is not provided (unsupervised analysis)
        num_classes = 1
        if y is not None:
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
        else:
            # If there is no class information the algorithm will iteratively group
            # samples into two classes based on minimum within class mean differences
            # For each feature, the expression levels in all samples are sorted in
            # descending order and then divided into two subsets
            if partition_data:
                y = self.__binary_partitioning(X=X)
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

        # Define the initial subsampling size for recurrent-sampling differential analysis
        num_subsamples = self.num_subsamples
        subsampling_size = self.subsampling_size
        if subsampling_size is None:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
                if np.min(temp) <= 2:
                    temp = int(np.min(temp))
                else:
                    temp = int(np.sqrt(np.min(temp)))
            else:
                temp = int(np.sqrt(num_examples))
            subsampling_size = temp
        else:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
                temp = np.min(temp)
            else:
                temp = num_examples
            if subsampling_size > temp:
                subsampling_size = temp

        if self.calculate_deltadisp or self.calculate_fisher:
            # Step 1: Iterative subsampling process to select and rank
            # significant features
            # Define IQR and P matrices
            P = np.zeros((num_features, num_subsamples))
            R = np.zeros((num_features,))
            if num_classes > 1:
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
                        delta_means = 0
                        if self.calculate_deltamean:
                            mean_i = np.mean(X[examples_i], axis=0)
                            mean_j = np.mean(X[examples_j], axis=0)
                            delta_means = np.absolute(mean_i - mean_j)
                            np.nan_to_num(delta_means, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        temp[:, combination_idx] += (delta_h + delta_means) / num_subsamples
                        if subsampling_size < 30:
                            _, pvalue = ttest_ind(X[examples_i], X[examples_j])
                        else:
                            _, pvalue = ztest(X[examples_i], X[examples_j])
                        P[:, sample_idx] += pvalue / num_combinations
                    combination_idx += 1
                R = np.max(temp, axis=1)
                if self.delta_type == "hvf":
                    del temp, disp1, disp2, delta_h
                else:
                    del temp, iq_range_i, iq_range_j, delta_h
            else:
                sample2example = np.zeros((num_subsamples, num_examples), dtype=np.int16)
                temp = np.zeros((num_features, num_subsamples))
                for sample_idx in range(num_subsamples):
                    subset = np.random.choice(a=num_examples, size=subsampling_size, replace=False)
                    delta_h = iqr(X[subset], axis=0, rng=self.iqr_range, scale=1.0)
                    P[:, sample_idx] = pvalue
                    sample2example[sample_idx, subset] = 1
                    temp[:, sample_idx] = np.absolute(delta_h)
                R = np.max(temp, axis=1)
            np.nan_to_num(R, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Check if iqr statistics for each feature has high variances across samples
            if num_classes < 2:
                P = 1 - norm.cdf(zscore(P, axis=1))
                P[P == np.inf] = 0
                np.nan_to_num(P, copy=False)

        # Step 2: Apply Fisher's method for combined probability
        if self.calculate_fisher:
            I = -2 * np.log(P)
            I[I == np.inf] = 0
            I = np.sum(I, axis=1)
            # Standardize I to be used in the final ranking of features. This is useful 
            # to calculate p-values)
            I = (I - 2 * num_subsamples) / np.sqrt(4 * num_subsamples)
            # Keep only the highest Fisher's statisitcs
            I[I < 0] = SEED_VALUE
            I = np.absolute(I)
            np.nan_to_num(I, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            # Calculate p-values from the Chi-square distribution with
            #  2 x self.subsets degrees of freedom
            # P = 1 - chi2.cdf(x = I, df = 2 * self.num_subsamples)
            # Adjusted p-values by BH method
            # _, P = fdrcorrection(pvals=P, alpha=0.05, is_sorted=False)
            del P

        # Step 3: Discriminative power calculation
        if self.calculate_disc_power:
            temp = []
            for class_idx in range(num_classes):
                examples_idx = np.where(y == class_idx)[0]
                temp.append(len(examples_idx))
            min_size = np.min(temp)
            slice_size = subsampling_size
            weight_range = self.weight_range
            O = np.zeros((num_features, 4))
            if self.bin_pvalues:
                O = np.zeros((num_features, 1))
            for feature_idx in range(num_features):
                if num_classes > 1:
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
                    if self.bin_pvalues:
                        O[feature_idx] = np.mean(temp_pvalues)
                        continue
                    # Complete change
                    if weight_range[0] > np.mean(temp_pvalues):  # or 0.1
                        O[feature_idx, 0] = 1
                    # Majority change
                    elif weight_range[1] > np.mean(temp_pvalues) >= weight_range[0]:  # or [0.4, 0.1]
                        O[feature_idx, 1] = 1
                    # Minority change
                    elif weight_range[2] > np.mean(temp_pvalues) >= weight_range[1]:  # or [0.8, 0.4]
                        O[feature_idx, 2] = 1
                    # Mixed change
                    else:
                        O[feature_idx, 3] = 1
                else:
                    temp_pvalues = 1 - norm.cdf(zscore(X[:, feature_idx]))
                    if self.bin_pvalues:
                        O[feature_idx] = np.mean(temp_pvalues)
                        continue
                    # Complete change
                    if 0.1 > np.mean(temp_pvalues):  # or 0.1
                        O[feature_idx, 0] = 1
                    # Majority change
                    elif 0.3 > np.mean(temp_pvalues) >= 0.1:  # or [0.4, 0.1]
                        O[feature_idx, 1] = 1
                    # Minority change
                    elif 0.5 > np.mean(temp_pvalues) >= 0.3:  # or [0.8, 0.4]
                        O[feature_idx, 2] = 1
                    # Mixed change
                    else:
                        O[feature_idx, 3] = 1
            if self.bin_pvalues:
                temp = KBinsDiscretizer(n_bins=len(self.feature_weight), encode="ordinal",
                                        strategy="uniform").fit_transform(O)
                O = np.zeros((num_features, len(self.feature_weight)), dtype=np.int8)
                for bin_idx in range(len(self.feature_weight)):
                    O[np.where(temp == bin_idx)[0], bin_idx] = 1
                del temp

        # Step 4: Estimating features statistics based on combined parameters (I, O, R)
        if self.calculate_deltadisp:
            R /= R.sum()
        else:
            R = 0
        if self.calculate_disc_power:
            O = self.feature_weight.dot(O.T)
        else:
            O = np.ones((num_features,))
        if self.calculate_fisher:
            I = np.multiply(I, O)
            I /= I.sum()
        else:
            I = O
            I /= I.sum()
        results = R + I

        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results
