'''
Hierarchical DeteCtion of celluLar hEterogeneity by anAlyzing variatioNs of cElls.
'''

from itertools import combinations

import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.stats import f_oneway, ks_2samp
from scipy.stats import iqr, ttest_ind, pearsonr
from scipy.stats import zscore, gamma
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import euclidean_distances
from statsmodels.stats.weightstats import ztest
from utility.utils import clustering

SEED_VALUE = 0.001


class HCLEANSE:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75), num_subsamples: int = 100,
                 subsampling_size: int = 3, significant_p: float = 0.05, partition_by_anova: bool = False,
                 feature_weight: list = [0.4, 0.3, 0.2, 0.1], calculate_hstatistic: bool = True,
                 num_components: int = 10, metric: str = "euclidean", num_subclusters: int = 10,
                 binary_clustering: bool = True, max_features: int = 50,
                 max_depth: int = 2, min_samples_split: int = 3, num_estimators: int = 5, num_rounds: int = 50,
                 calculate_pval: bool = False, num_jobs: int = 2):
        """
        Detect a set of features that best describe clusters and their subclusters.

        In most cases, `robust` normalization is a better approach. For efficiency 
        reasons, both `max_depth` and `num_estimators` paramters should be kept at a 
        convinvent level depending on data. It is highly recommend to avoid computing
        permutation test by enabling the `calculate_pval` paramters with high 
        dimensional datasets.

        Parameters
        ----------
        normalize : {array-like, sparse matrix} of shape (n_samples_X, n_features)
            An array where each row is a sample and each column is a feature.

        q : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
                default=None
            An array where each row is a sample and each column is a feature.
            If `None`, method uses `Y=X`.

        iqr_range : array-like of shape (n_samples_Y,) or (n_samples_Y, 1) \
                or (1, n_samples_Y), default=None
            Pre-computed dot-products of vectors in Y (e.g.,
            ``(Y**2).sum(axis=1)``)
            May be ignored in some cases, see the note below.

        squared : bool, default=False
            Return squared Euclidean distances.

        X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1) \
                or (1, n_samples_X), default=None
            Pre-computed dot-products of vectors in X (e.g.,
            ``(X**2).sum(axis=1)``)
            May be ignored in some cases, see the note below.

        Attributes
        ----------
        affinity_matrix_ : array-like of shape (n_samples, n_samples)
            Affinity matrix used for clustering. Available only after calling
            ``fit``.

        labels_ : ndarray of shape (n_samples,)
            Labels of each point

        n_features_in_ : int
            Number of features seen during :term:`fit`.

            .. versionadded:: 0.24

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

            .. versionadded:: 1.0
        """

        self.normalize = normalize  # robust or zscore (default: None)
        self.q = q
        self.iqr_range = iqr_range
        self.num_subsamples = num_subsamples
        self.subsampling_size = subsampling_size
        self.significant_p = significant_p
        self.partition_by_anova = partition_by_anova
        self.num_components = num_components
        self.metric = metric
        self.num_subclusters = num_subclusters
        self.binary_clustering = binary_clustering
        # Weighting four feature profiles
        if len(feature_weight) > 4 or len(feature_weight) == 0:
            feature_weight = [0.4, 0.3, 0.2, 0.1]
        self.feature_weight = np.array(feature_weight) / np.sum(feature_weight)
        self.calculate_hstatistic = calculate_hstatistic
        # The following parameters are applied to detect subclasses
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # This boolean argument indicates whether to perform permutation tests or not
        self.calculate_pval = calculate_pval
        # The maximum number of permutation tests rounds
        self.num_rounds = num_rounds
        # The number of trees to construct
        self.num_estimators = num_estimators
        # The maximum number of concurrently running jobs
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
            cluster_type = "spectral"
            num_neighbors = num_examples // 2
            if num_examples <= 5:
                cluster_type = "kmeans"
            y = clustering(X=K.T, cluster_type=cluster_type, affinity="nearest_neighbors",
                           num_neighbors=num_neighbors, num_clusters=2, num_jobs=self.num_jobs,
                           predict=True)
        return y

    def __recursive_split(self, X, y, node, depth: int, features: list, node_idx: list,
                          terminal_node_prob: list):
        '''
        Recursively apply this function to create splits for a node (node) or make
        terminal with specified depth (depth).
        :type node: dict
        :param node: A child or root node that might have zero children, one child
            (one side makes a prediction directly) or two child nodes (left and right).
            The node comprises of: "label0" for the class label associated with the
            left child, "label1" for the class label associated with the right child,
            "splitVariable" for the feature used in splitting, and "node_data" that
            holds data for both right and left children.
        :type depth: int
        :param depth: The depth of the tree that is recursively built.
        '''
        # The two groups of data split by the node are extracted for use and deleted
        # from the node. As we work on these groups the node no longer requires access
        # to these data.
        left, right = node['node_data']
        del node['node_data']

        # Check if either left or right group of data is empty and if so we create a
        # terminal node using what samples we posses.
        if len(left) == 0 or len(right) == 0:
            node['left'] = node['right'] = np.concatenate((right, left)).tolist()
            node['cluster_level'] = True
            return

        # Check if the tree reached the maximum depth and if so we create a terminal
        # node.
        if depth >= self.max_depth:
            node['left'] = node['right'] = np.concatenate((right, left)).tolist()
            node['cluster_level'] = True
            return

        # Check if the size of examples for each classes is above 2 for recurrent-sampling
        # differential analysis
        for samples_idx in [left, right]:
            for class_idx in np.unique(y[samples_idx]):
                if len(y[samples_idx] == class_idx) < 2:
                    node['left'] = node['right'] = np.concatenate((right, left)).tolist()
                    node['cluster_level'] = True
                    return

        # Process the right child by either creating a terminal node if the group of
        # data is too small, otherwise creating and adding the right node in a depth
        # first manner until the bottom of the tree is reached on this branch.
        if len(right) <= self.min_samples_split:
            node['right'] = right
            node['cluster_level'] = True
        else:
            temp = self.__fit_predict(X=X[right][:, features], y=y[right])
            split_features, samples_left, samples_right, features_statistics, class_distribution = temp
            node_idx.append(len(node_idx))
            node['right'] = {
                'node_data': (np.array(right)[samples_left].tolist(), np.array(right)[samples_right].tolist()),
                'node_idx': len(node_idx),
                'split_features': np.array(features)[split_features].tolist(),
                'features_statistics': features_statistics,
                'class_distribution': class_distribution}
            self.__recursive_split(X=X, y=y, node=node['right'], depth=depth + 1,
                                   features=np.array(features)[split_features].tolist(), node_idx=node_idx,
                                   terminal_node_prob=terminal_node_prob)

        # Process the left child by either creating a terminal node if the group of
        # data is too small, otherwise creating and adding the left node in a depth
        # first manner until the bottom of the tree is reached on this branch.
        if len(left) <= self.min_samples_split:
            node['left'] = left
            node['cluster_level'] = True
        else:
            temp = self.__fit_predict(X=X[left][:, features], y=y[left])
            split_features, samples_left, samples_right, features_statistics, class_distribution = temp
            node_idx.append(len(node_idx))
            node['left'] = {
                'node_data': (np.array(left)[samples_left].tolist(), np.array(left)[samples_right].tolist()),
                'node_idx': len(node_idx),
                'split_features': np.array(features)[split_features].tolist(),
                'features_statistics': features_statistics,
                'class_distribution': class_distribution}
            self.__recursive_split(X=X, y=y, node=node['left'], depth=depth + 1,
                                   features=np.array(features)[split_features].tolist(), node_idx=node_idx,
                                   terminal_node_prob=terminal_node_prob)

    def __traverse_features(self, tree):
        # Check whether the tree is a type of dictionary
        result = list()
        if isinstance(tree, dict):
            result = self.__traverse_features(tree["right"])
            if "cluster_level" in tree:
                result.append([tree["features_statistics"], tree["split_features"], "cluster_level"])
            elif "root" in tree:
                result.append([tree["features_statistics"], tree["split_features"], "root"])
            result = result + self.__traverse_features(tree["left"])
        return result

    def __traverse_samples(self, tree):
        # Check whether the tree is a type of dictionary
        result = list()
        if isinstance(tree, dict):
            result = self.__traverse_samples(tree["right"])
            if isinstance(tree["right"], list):
                result.append(tree["right"])
            result = result + self.__traverse_samples(tree["left"])
        return result

    def __fit_predict(self, X, y):
        # Extract the number of samples and the number of features.
        num_examples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Binary partition data is the number of groups is less than 2 
        if num_classes == 1:
            y = self.__binary_partitioning(X=X)

        # Each group should have at least two examples
        for class_idx in np.unique(y):
            if len(np.where(y == class_idx)[0]) < 2:
                temp = np.random.choice(a=np.where(y != class_idx)[0], size=1, replace=False)
                y[temp] = class_idx
        num_classes = len(np.unique(y))

        # Total number of combinations of groups
        num_combinations = len(list(combinations(range(num_classes), 2)))

        # Define the subsampling size for recurrent-sampling differential analysis
        num_subsamples = self.num_subsamples
        subsampling_size = self.subsampling_size
        temp = list()
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp.append(len(examples_idx))
        if subsampling_size is None:
            min_size = int(np.sqrt(np.min(temp)))
        else:
            min_size = int(np.min(temp))
            if self.subsampling_size <= min_size:
                min_size = self.subsampling_size
        if min_size < 2:
            min_size = int(np.min(temp))
        subsampling_size = min_size

        # Step 1: Recurrent-sampling differential analysis to select and rank
        # significant features
        # Define frequency A and raw p-value P matrices
        A = np.zeros((num_features, num_examples)) + SEED_VALUE
        P = np.zeros((num_features, num_subsamples))
        R = np.zeros((num_features,))
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
                feature_pvalue = np.where(pvalue <= self.significant_p)[0]
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

        # Apply Fisher's method for combined probability
        I = -2 * np.log(P)
        I[I == np.inf] = 0
        I = np.sum(I, axis=1)
        # Standardize I to be used in the final ranking of features. This is useful
        # to calculate p-values)
        I = (I - 2 * num_subsamples) / np.sqrt(4 * num_subsamples)
        # Keep only the highest Fisher's statistics
        I[I < 0] = SEED_VALUE
        I = np.absolute(I)
        del P

        # Step 2: Non-negative Matrix Factorization (NMF) using frequency matrices
        nmf = NMF(n_components=self.num_components, l1_ratio=0, max_iter=1000, random_state=12345)
        W = nmf.fit_transform(X=A)
        H = nmf.components_.T
        del nmf

        # Step 3: Define binary clusters of samples
        if self.metric == "euclidean":
            temp = euclidean_distances(X=H, Y=H)
        else:
            temp = np.zeros((num_examples, num_examples))
            for i in range(num_examples):
                for j in range(i + 1, num_examples):
                    temp[i, j] = pearsonr(x=H[i], y=H[j])[0]
            temp = temp + temp.T
        subclusters = clustering(X=temp, cluster_type="spectral", affinity="nearest_neighbors", num_neighbors=5,
                                 num_clusters=2, num_jobs=self.num_jobs, predict=True)
        right = np.where(subclusters == 1)[0]
        left = np.where(subclusters == 0)[0]

        # Step 4: Mapping the NMF data for features and samples in a multidimensional space
        E = euclidean_distances(X=W, Y=H)
        # Estimate gene-wise dispersion
        D = np.zeros((num_features, num_examples))
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp = zscore(X[examples_idx])
            D[:, examples_idx] = temp.T
        # Compute the heterogeneity statistic of each profile
        H_statistic = np.multiply(D, E) - np.mean(np.multiply(D, E), axis=1)[:, None]
        del examples_idx, D, E, W, H

        # Step 5: Identification of 4 feature profiles
        O = np.zeros((num_features, 4))
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
            if 0.1 > np.mean(temp_lst):
                O[feature_idx, 0] = 1
            # Majority change
            elif 0.4 > np.mean(temp_lst) >= 0.1:
                O[feature_idx, 1] = 1
            # Minority change
            elif 0.8 > np.mean(temp_lst) >= 0.4:
                O[feature_idx, 2] = 1
            # Mixed change
            else:
                O[feature_idx, 3] = 1

        # Step 6: Calculate new H statistics based on absolute differences between
        # pairwise class of precomputed H-statistics
        new_H = np.zeros((num_features, num_classes))
        for class_idx in range(num_classes):
            examples_idx = np.where(y == class_idx)[0]
            temp = np.mean(np.absolute(H_statistic[:, examples_idx]), axis=1)
            new_H[:, class_idx] = temp
        for i, j in combinations(range(num_classes), 2):
            new_H[:, 0] += np.absolute(new_H[:, i] - new_H[:, j])
        H_statistic = new_H[:, 0]
        del new_H

        # Step 7: Feature ranking based on combined parameters (I, O, R, H)
        new_cost = np.multiply(I, self.feature_weight.dot(O.T)) + np.multiply(R, H_statistic)
        new_cost = np.multiply(I, self.feature_weight.dot(O.T)) + R
        np.nan_to_num(new_cost, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Return the follwings
        if new_cost.shape[0] >= 2:
            shape, loc, scale = gamma.fit(zscore(new_cost))
            temp = 1 - gamma.cdf(zscore(new_cost), shape, loc=loc, scale=scale)
            selected_features = np.where(temp <= self.significant_p)[0]
            if len(selected_features) == 0 or len(selected_features) > self.max_features:
                selected_features = np.argsort(temp)
                selected_features = selected_features[:self.max_features]
            split_features = selected_features.tolist()
        else:
            split_features = range(num_features)
        features_statistics = new_cost[split_features]
        class_distribution = {"1": len(right) / num_examples, "0": len(left) / num_examples}
        samples_left = left.tolist()
        samples_right = right.tolist()

        return split_features, samples_left, samples_right, features_statistics, class_distribution

    def fit_predict(self, X, y=None, control_class: int = 0, case_class: int = 1, return_best_features: bool = True,
                    return_clusters: bool = True):
        # Extract properties
        num_examples, num_features = X.shape

        # Set maximum number of features to be used for computing the best split
        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(num_features)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(num_features)))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = num_features
        self.max_features = max_features

        # Check if classes information is not provided (unsupervised analysis)
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
        else:
            # If there is no class information the algorithm will iteratively group
            # samples into two classes based on minimum within class mean differences
            # For each feature, the expression levels in all samples are sorted in
            # descending order and then divided into two subsets
            y = self.__binary_partitioning(X=X)
        num_classes = len(np.unique(y))

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
        if self.subsampling_size is None:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
                temp = int(np.sqrt(np.min(temp)))
            else:
                temp = int(np.sqrt(num_examples))
            self.subsampling_size = temp
        else:
            if num_classes > 1:
                temp = list()
                for class_idx in range(num_classes):
                    examples_idx = np.where(y == class_idx)[0]
                    temp.append(len(examples_idx))
                temp = np.min(temp)
            else:
                temp = num_examples
            if self.subsampling_size > temp:
                self.subsampling_size = temp

        # Hierarchical partitioning based on tree decision
        forget_rate = 0.5
        learning_rate = 0.01
        # Store constructed trees
        list_estimators = list()
        for estimator_idx in range(self.num_estimators):
            estimator = self.__fit_predict(X=X, y=y)
            split_features, samples_left, samples_right, features_statistics, class_distribution = estimator
            del estimator
            # The right child holds the data based on the present of the feature used
            # for splitting while the left child holds the data based on the absent of the
            # feature used for splitting
            estimator = {'node_data': (samples_left, samples_right),
                         'node_idx': 0,
                         'root': True,
                         'split_features': split_features,
                         'features_statistics': features_statistics,
                         'class_distribution': class_distribution}
            # Construct the decision tree.
            terminal_node_prob = []
            self.__recursive_split(X=X, y=y, node=estimator, depth=1, features=split_features, node_idx=list(),
                                   terminal_node_prob=terminal_node_prob)
            list_estimators.append(estimator)

        results = None
        best_cost = 0
        best_features = None
        best_estimator = list_estimators[0]
        features_statistics = np.zeros((num_features,))
        split_features = list()
        for estimator in list_estimators:
            temp = self.__traverse_features(tree=estimator)
            for fs, sf, t in temp:
                features_statistics[sf] += fs
                if t != "root":
                    split_features.extend(sf)
            # features_statistics = np.sum(list(zip(*temp))[0], axis=0)
            # split_features = np.unique(np.concatenate(list(zip(*temp))[1])).tolist()
            split_features = np.unique(split_features).tolist()
            tree_cost = np.sum(features_statistics[split_features])
            if best_cost < tree_cost:
                best_cost = tree_cost
                results = features_statistics
                best_features = split_features
                best_estimator = estimator
        del list_estimators, features_statistics
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.calculate_pval and num_classes > 1:
            # Total number of combinations of groups
            num_combinations = len(list(combinations(range(num_classes), 2)))
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

        temp = list()
        if return_best_features or return_clusters:
            temp.append(results)
        if return_best_features:
            temp.append(best_features)
        if return_clusters:
            cluster2samples = self.__traverse_samples(tree=best_estimator)
            clusters = np.zeros((num_examples,), dtype=np.int8)
            for cluster_idx, item in enumerate(cluster2samples):
                clusters[item] = cluster_idx
            temp.append(clusters.tolist())
        results = temp
        return results

