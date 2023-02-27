'''
DeteCtion of celluLar hEterogeneity by anAlyzing variatioNs of cElls.
'''

from itertools import combinations

import numpy as np
from mlxtend.evaluate import permutation_test
from prince import CA
from scipy.stats import f_oneway, ks_2samp
from scipy.stats import iqr, norm, zscore, ttest_ind
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.weightstats import ztest
from utility.utils import clustering

SEED_VALUE = 0.001


class PHeT:
    def __init__(self, normalize: str = "zscore", iqr_range: int = (25, 75), num_subsamples: int = 1000,
                 subsampling_size: int = None, alpha_subsample: float = 0.05, partition_by_anova: bool = False,
                 calculate_deltaiqr: bool = True, calculate_fisher: bool = True, calculate_profile: bool = True,
                 calculate_hstatistic: bool = False, num_components: int = 10, num_subclusters: int = 10,
                 binary_clustering: bool = True, bin_KS_pvalues: bool = False, feature_weight: list = None,
                 weight_range: list = None, permutation_test: bool = False, num_rounds: int = 10000,
                 num_jobs: int = 2):
        self.normalize = normalize  # robust or zscore
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.alpha_subsample = alpha_subsample
        self.partition_by_anova = partition_by_anova
        self.calculate_deltaiqr = calculate_deltaiqr
        self.calculate_fisher = calculate_fisher
        self.calculate_profile = calculate_profile
        if calculate_deltaiqr and calculate_fisher and calculate_profile:
            self.calculate_deltaiqr = True
            self.calculate_fisher = True
            self.calculate_profile = True
        self.calculate_hstatistic = calculate_hstatistic
        self.num_components = num_components
        self.num_subclusters = num_subclusters
        self.binary_clustering = binary_clustering
        self.bin_KS_pvalues = bin_KS_pvalues
        if len(feature_weight) < 2:
            feature_weight = [0.4, 0.3, 0.2, 0.1]
        if not bin_KS_pvalues:
            if len(feature_weight) > 4:
                feature_weight = [0.4, 0.3, 0.2, 0.1]
            if len(weight_range) != 3:
                weight_range = [0.1, 0.3, 0.5]
        self.feature_weight = np.array(feature_weight) / np.sum(feature_weight)
        self.weight_range = weight_range  # [0.1, 0.4, 0.8]
        self.permutation_test = permutation_test
        self.num_rounds = num_rounds
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

        # Define the initial subsampling size for recurrent-sampling differential analysis
        num_subsamples = self.num_subsamples
        subsampling_size = self.subsampling_size
        if subsampling_size is None:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
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

        if self.calculate_deltaiqr or self.calculate_fisher or self.calculate_hstatistic:
            # Step 1: Recurrent-sampling differential analysis to select and rank
            # significant features
            # Define frequency A and raw p-value P matrices
            A = np.zeros((num_features, num_examples)) + SEED_VALUE
            P = np.zeros((num_features, num_subsamples))
            R = np.zeros((num_features,))
            if num_classes > 1:
                # Make transposed matrix with shape (feat per class, observation per class)
                # find mean and iqr difference between genes
                temp = np.zeros((num_features, num_combinations))
                combination_idx = 0
                for i, j in combinations(range(num_classes), 2):
                    for sample_idx in range(num_subsamples):
                        examples_i = np.where(y == i)[0]
                        examples_j = np.where(y == j)[0]
                        examples_i = np.random.choice(a=examples_i, size=subsampling_size, replace=False)
                        examples_j = np.random.choice(a=examples_j, size=subsampling_size, replace=False)
                        iq_range_i = iqr(X[examples_i], axis=0, rng=self.iqr_range, scale=1.0)
                        iq_range_j = iqr(X[examples_j], axis=0, rng=self.iqr_range, scale=1.0)
                        iq_range = iq_range_i - iq_range_j
                        temp[:, combination_idx] += np.absolute(iq_range) / num_subsamples
                        if subsampling_size < 30:
                            _, pvalue = ttest_ind(X[examples_i], X[examples_j])
                        else:
                            _, pvalue = ztest(X[examples_i], X[examples_j])
                        P[:, sample_idx] += pvalue / num_combinations
                        feature_pvalue = np.where(pvalue <= self.alpha_subsample)[0]
                        if len(feature_pvalue) > 0:
                            up = feature_pvalue[np.where(iq_range[feature_pvalue] > 0)[0]]
                            down = feature_pvalue[np.where(iq_range[feature_pvalue] <= 0)[0]]
                            if len(up) > 0:
                                for idx in examples_i:
                                    A[up, idx] += 1
                            if len(down) > 0:
                                for idx in examples_j:
                                    A[down, idx] += 1
                    combination_idx += 1
                R = np.max(temp, axis=1)
                del temp, feature_pvalue, up, down
            else:
                sample2example = np.zeros((num_subsamples, num_examples), dtype=np.int16)
                temp = np.zeros((num_features, num_subsamples))
                for sample_idx in range(num_subsamples):
                    subset = np.random.choice(a=num_examples, size=subsampling_size, replace=False)
                    iq_range = iqr(X[subset], axis=0, rng=self.iqr_range, scale=1.0)
                    P[:, sample_idx] = pvalue
                    sample2example[sample_idx, subset] = 1
                    temp[:, sample_idx] = np.absolute(iq_range)
                R = np.max(temp, axis=1)
            np.nan_to_num(R, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Check if iqr statistics for each feature has high variances across samples
            if num_classes < 2:
                P = 1 - norm.cdf(zscore(P, axis=1))
                P[P == np.inf] = 0
                np.nan_to_num(P, copy=False)
                for feature_idx in range(num_features):
                    samples_idx = np.where(P[feature_idx] < self.alpha_subsample)[0]
                    for sample_idx in samples_idx:
                        examples_idx = np.nonzero(sample2example[sample_idx])[0]
                        A[feature_idx, examples_idx] += 1

        if self.calculate_fisher:
            # Step 2: Apply Fisher's method for combined probability
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

        if self.calculate_profile:
            # Step 3: Identification of 4 feature profiles
            weight_range = self.weight_range
            O = np.zeros((num_features, 4))
            if self.bin_KS_pvalues:
                O = np.zeros((num_features, 1))
            for feature_idx in range(num_features):
                if num_classes > 1:
                    temp_pvalues = list()
                    for i, j in combinations(range(num_classes), 2):
                        examples_i = np.where(y == i)[0]
                        examples_j = np.where(y == j)[0]
                        examples_i = X[examples_i, feature_idx]
                        examples_j = X[examples_j, feature_idx]
                        pvalue = ks_2samp(examples_i, examples_j)[1]
                        temp_pvalues.append(pvalue)
                    if self.bin_KS_pvalues:
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
                    if self.bin_KS_pvalues:
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
            # TODO: define gradient bounds and a boolean indicator
            # O = (.1 - .4) * (((O - np.min(O)) / (np.max(O) - np.min(O)))) + .4
            if self.bin_KS_pvalues:
                temp = KBinsDiscretizer(n_bins=len(self.feature_weight), encode="ordinal",
                                        strategy="uniform").fit_transform(O)
                O = np.zeros((num_features, len(self.feature_weight)), dtype=np.int8)
                for bin_idx in range(len(self.feature_weight)):
                    O[np.where(temp == bin_idx)[0], bin_idx] = 1
                del temp

        if self.calculate_hstatistic:
            # Step 3.1: Correspondence analysis (CA) using frequency matrices
            ca = CA(n_components=self.num_components, n_iter=self.num_rounds, benzecri=False)
            ca.fit(X=A)

            # Step 3.2: Mapping the CA data for features and samples in a multidimensional space 
            E = euclidean_distances(X=ca.U_, Y=ca.V_.T)
            # Estimate gene-wise dispersion
            D = np.zeros((num_features, num_examples))
            if num_classes > 1:
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp = zscore(X[examples_idx])
                    # temp = (X[examples_idx] - np.mean(X[examples_idx], axis=0)) ** 2
                    D[:, examples_idx] = temp.T
            else:
                D = zscore(X, axis=0).T
            # Compute the heterogeneity statistic of each profile
            H = np.multiply(D, E) - np.mean(np.multiply(D, E), axis=1)[:, None]
            del examples_idx, D, E

            # Step 3.3: Calculate new H statistics based on absolute differences between 
            # pairwise class of precomputed H-statistics
            if num_classes > 1:
                new_H = np.zeros((num_features, num_classes))
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp = np.mean(np.absolute(H[:, examples_idx]), axis=1)
                    new_H[:, class_idx] = temp
                for i, j in combinations(range(num_classes), 2):
                    new_H[:, 0] += np.absolute(new_H[:, i] - new_H[:, j])
                H = new_H[:, 0]
                del new_H
            else:
                H = np.mean(H, axis=1)
            np.nan_to_num(H, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if self.calculate_deltaiqr:
                R = np.multiply(R, H)
            else:
                R = H
            del H
        A = 0

        # Step 4: Estimating features statistics based on combined parameters (I, O, R, H)
        if self.calculate_deltaiqr:
            R /= R.sum()
        else:
            R = 0
        if self.calculate_profile:
            O = self.feature_weight.dot(O.T)
        else:
            O = np.ones((num_features,))
        if self.calculate_fisher:
            I = np.multiply(I, O)
            I /= I.sum()
        else:
            I = O
            I /= I.sum()
        H = R + I
        np.nan_to_num(H, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.permutation_test and num_classes > 1:
            # Permutation based p-value calculation using approximate method
            pvals = np.zeros((num_features,))
            for feature_idx in range(num_features):
                for i, j in combinations(range(num_classes), 2):
                    examples_i = np.where(y == i)[0]
                    examples_j = np.where(y == j)[0]
                    examples_i = X[examples_i, feature_idx]
                    examples_j = X[examples_j, feature_idx]
                    if self.direction == "up":
                        temp = permutation_test(x=examples_i, y=examples_j, func="x_mean > y_mean",
                                                method="approximate", num_rounds=self.num_rounds)
                    elif self.direction == "down":
                        temp = permutation_test(x=examples_i, y=examples_j, func="x_mean < y_mean",
                                                method="approximate", num_rounds=self.num_rounds)
                    else:
                        temp = permutation_test(x=examples_i, y=examples_j, func="x_mean != y_mean",
                                                method="approximate", num_rounds=self.num_rounds)
                    pvals[feature_idx] += temp / num_combinations
            H = np.vstack((H, pvals)).T
        else:
            H = np.reshape(H, (H.shape[0], 1))

        results = H
        return results
