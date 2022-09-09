'''
TODO: 
1. Borrow ideas from MOST wrt Order Statistics
2. Link those methods with Outliers detection
3. unravelinG ceLlular heterOgeneity By intra-cellULar varIatioN	
'''

from itertools import combinations

import numpy as np
from mlxtend.evaluate import permutation_test
from prince import CA
from scipy.stats import iqr, zscore, ttest_ind
from scipy.stats import pearsonr, f_oneway, ks_2samp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from utility.utils import clustering


class GLOBULIN:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75), num_subsamples: int = 100,
                 subsampling_size: int = 3, significant_p: float = 0.05, partition_by_anova: bool = False,
                 num_components: int = 10, num_subclusters: int = 10, binary_clustering: bool = True,
                 calculate_pval: bool = False, num_rounds: int = 50, num_jobs: int = 2):
        self.normalize = normalize  # robust or zscore (default: None)
        self.q = q
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.significant_p = significant_p
        self.partition_by_anova = partition_by_anova
        self.num_components = num_components
        self.num_subclusters = num_subclusters
        self.binary_clustering = binary_clustering
        self.calculate_pval = calculate_pval
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
                           num_clusters=2, num_jobs=2, predict=True)
        return y

    def fit_predict(self, X, y=None, control_class: int = 0, case_class: int = 1):
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
            partition_data = False
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

        # Step 1: Recurrent-sampling differential analysis to select and rank
        # significant features
        if self.subsampling_size is None:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
                temp = np.sqrt(np.min(temp)).astype(int)
            else:
                temp = np.sqrt(num_examples).astype(int)
            self.subsampling_size = temp
        # Define frequency A and raw p-value P matrices
        A = np.zeros((num_features, num_examples))
        P = np.zeros((num_features, self.num_subsamples))
        # if num_classes == 2:
        #     A = np.zeros((2 * num_features, num_examples))
        A = A + 0.001
        IQR = np.zeros((num_features, self.num_subsamples))
        # TODO: find a way to cluster samples into several groups and then
        #  iteratively partition samples (hierarchical partition)
        for sample_idx in range(self.num_subsamples):
            temp_iq = list()
            if num_classes > 1:
                # Make transposed matrix with shape (feat per class, observation per class)
                # find mean and iqr difference between genes
                temp = np.zeros((num_features, num_combinations))
                combination_idx = 0
                for i, j in combinations(range(num_classes), 2):
                    examples_i = np.where(y == i)[0]
                    examples_i = np.random.choice(a=examples_i, size=self.subsampling_size, replace=False)
                    examples_j = np.where(y == j)[0]
                    examples_j = np.random.choice(a=examples_j, size=self.subsampling_size, replace=False)
                    for feature_idx in range(num_features):
                        iq_range = iqr(X[examples_i, feature_idx], rng=self.iqr_range, scale=1.0)
                        iq_range = iq_range - iqr(X[examples_j, feature_idx], rng=self.iqr_range, scale=1.0)
                        _, pvalue = ttest_ind(X[examples_i, feature_idx], X[examples_j, feature_idx])
                        if pvalue <= self.significant_p:
                            P[feature_idx, sample_idx] = pvalue
                            if iq_range > 0:
                                A[feature_idx, examples_i] += 1
                            else:
                                A[feature_idx, examples_j] += 1
                        temp[feature_idx, combination_idx] = iq_range
                    combination_idx += 1
                temp_iq = np.max(np.absolute(temp), axis=1)
            else:
                subset = np.random.choice(a=num_examples, size=self.subsampling_size, replace=False)
                for feature_idx in range(num_features):
                    iq_range = iqr(X[subset, feature_idx], rng=self.iqr_range, scale=1.0)
                    if iq_range > 0:
                        A[feature_idx, subset] += 1
                        P[feature_idx, sample_idx] = iq_range
                    temp_iq.append(np.absolute(iq_range))
            IQR[:, sample_idx] = temp_iq
        del temp, temp_iq
        IQR = np.mean(IQR, axis=1)

        # Apply Fisher's method for combined probability
        if num_classes < 2:
            P /= np.sum(axis=1)
            P[P == np.inf] = 0
            np.nan_to_num(P, copy=False)
        X_f = -2 * np.log(P)
        X_f[X_f == np.inf] = 0
        X_f = np.sum(X_f, axis=1)
        del P
        # TODO: Calculate p-values from the Chi-square distribution with
        #  2 x self.subsets degrees of freedom
        # P = 1 - chi2.cdf(x = X_f, df = 2 * self.subsamples)
        # # Adjusted p-values by BH method
        # _, P = fdrcorrection(pvals=P, alpha=0.05, is_sorted=False)
        # Standardize X_f to be used in the final ranking of features
        X_f = (X_f - 2 * self.num_subsamples) / np.sqrt(4 * self.num_subsamples)

        # Step 2: Correspondence analysis (CA) using frequency matrices
        ca = CA(n_components=self.num_components, n_iter=self.num_rounds, benzecri=False)
        ca.fit(X=A)

        # Step 3: Mapping the CA data for features and samples in a multidimensional space 
        P = euclidean_distances(X=ca.U_, Y=ca.V_.T)
        # Estimate gene-wise dispersion
        D = np.zeros((num_features, num_examples))
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp = X[examples_idx] - np.mean(X[examples_idx], axis=0)
            D[:, examples_idx] = temp.T
        # Compute the heterogeneity statistic of each profile
        H = np.multiply(D, P) - np.mean(np.multiply(D, P), axis=1)[:, None]
        del examples_idx, temp, D, P

        # TODO: Step 4: Finding all classes and groups in the sample set (optional)
        discover_groups = False
        if discover_groups:
            C = np.zeros((num_examples, num_examples))
            for i in range(num_examples):
                for j in range(i + 1, num_examples):
                    C[i, j] = pearsonr(x=H[:, i], y=H[:, j])[0]
            C = C + C.T
            # TODO: Find an optimal number of k clusters-subclasses using hierarchical approach
            subclusters = self.num_subclusters
            temp_scores = np.zeros((self.num_subclusters - 2))
            for cluster_size in range(2, subclusters):
                model = AgglomerativeClustering(n_clusters=cluster_size)
                model.fit(X=C)
                temp = silhouette_score(X=C, labels=model.labels_)
                temp_scores[cluster_size - 2] = temp
            subclusters = np.argmax(temp_scores) + 2
            subclusters = clustering(X=C, cluster_type="agglomerative", affinity="euclidean", num_neighbors=5,
                                     num_clusters=subclusters, num_jobs=self.num_jobs, predict=True)

        # Step 5: Identification of 4 feature profiles
        O = np.zeros((num_features, 4))
        if num_classes > 1:
            for feature_idx in range(num_features):
                temp_lst = list()
                for i, j in combinations(range(num_classes), 2):
                    examples_i = np.where(y == i)[0]
                    examples_j = np.where(y == j)[0]
                    examples_i = X[examples_i, feature_idx]
                    examples_j = X[examples_j, feature_idx]
                    pvalue = ks_2samp(examples_i, examples_j)[1]
                    temp_lst.append(pvalue)
                # Complete change
                if 0.2 > np.mean(temp_lst):
                    O[feature_idx, 0] = 1
                # Majority change
                elif 0.4 > np.mean(temp_lst) >= 0.2:
                    O[feature_idx, 1] = 1
                # Minority change
                elif 0.8 > np.mean(temp_lst) >= 0.4:
                    O[feature_idx, 2] = 1
                # Mixed change
                else:
                    O[feature_idx, 3] = 1

        # Step 6: Feature ranking based on combined parameters
        # Three main parameters are calculated: (i) X_f, which highlights the most significant 
        # and constant differential changes among samples; (ii) H, which indicates how discriminant 
        # each feature is, given the sample subclasses; and (iii) both of O and standard deviation of raw
        # omic signal in each differential feature, assessing the variability along samples
        # to allow finding the most stable features that will be considered the best markers
        # for the classes or subclasses found
        change_weights = np.array([0.4, 0.3, 0.2, 0.1])
        new_H = np.zeros((num_features, num_classes))
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp = np.median(np.absolute(H[:, examples_idx]), axis=1)
            new_H[:, class_idx] = temp
        H = np.mean(new_H, axis=1)
        del new_H
        results = np.absolute(X_f) + IQR + H
        results += np.multiply(change_weights.dot(O.T), np.std(X.T, axis=1))

        if self.calculate_pval and num_classes > 1:
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
            results = np.vstack((results, pvals)).T
        else:
            results = np.reshape(results, (results.shape[0], 1))

        return results


if __name__ == "__main__":
    num_examples = 50
    num_features = 200
    control_class = 0
    case_class = 1
    y = np.random.randint(0, 2, num_examples)
    X = np.zeros((num_examples, num_features))
    temp = np.where(y == 0)[0]
    X[temp] = np.random.normal(size=(len(temp), num_features))
    temp = np.where(y == 1)[0]
    X[temp] = np.random.normal(loc=2, scale=5, size=(len(temp), num_features))
    model = GLOBULIN(normalize="robust")
    model.fit_predict(X=X, y=None, control_class=control_class,
                      case_class=case_class)
