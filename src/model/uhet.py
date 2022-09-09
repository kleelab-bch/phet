'''
TODO: 
1. Borrow ideas from MOST wrt Order Statsitics
2. Link those methods with Outliers detection
3. Perform multiple permuation tests
'''

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import iqr
from scipy.stats import zscore


class UHeT:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75),
                 calculate_pval: bool = True, num_iterations: int = 10000):
        self.normalize = normalize
        self.q = q
        self.iqr_range = iqr_range
        self.calculate_pval = calculate_pval
        self.num_iterations = num_iterations

    def fit_predict(self, X, y):
        """
        Hetero-Net Function

        Perform Deep Metric Learning with UMAP-based clustering to find subpopulations of classes

        Read more in the USER GUIDE

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples

        standardize : bool, default=True
            Standardizes data using zscore (NOTE: test more?)

        Attributes
        ----------
        features_ : list
            The features found by the algorithm. Ranked in order of importance

        score_ : list
            The score found by the algorithm for each feature


        References
        ----------
        NOTE: MINE


        Examples
        ----------
        NOTE: TODO
        """

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

        # make transposed matrix with shape (feat per class, observation per class)
        # find mean and iqr difference between genes
        diff_iqrs = list()
        diff_means = list()
        ttest_statistics = list()
        classes = list()
        for feature_idx in range(num_features):
            iqrs = list()
            means = list()
            ttests = list()
            for i in range(num_classes):
                examples_i = np.where(y == i)[0]
                for j in range(i + 1, num_classes):
                    examples_j = np.where(y == j)[0]
                    iqr1 = iqr(X[examples_i, feature_idx], rng=self.iqr_range, scale=1.0)
                    iqr2 = iqr(X[examples_j, feature_idx], rng=self.iqr_range, scale=1.0)
                    mean1 = np.mean(X[examples_i, feature_idx])
                    mean2 = np.mean(X[examples_j, feature_idx])
                    statistic = stats.ttest_ind(X[examples_i, feature_idx], X[examples_j, feature_idx])[0]
                    # # TODO: COMMENT if didnt work
                    # if np.sign(mean1) < np.sign(iqr1):
                    #     iqr_rng1 = np.mean(self.iqr_range).astype(int)
                    #     percentile1 = np.percentile(X[examples_i, feature_idx], iqr_rng1)
                    #     percentile2 = np.percentile(X[examples_i, feature_idx], self.iqr_range[1])
                    #     picked_examples = np.where(np.logical_and(X[examples_i, feature_idx] >= percentile1,
                    #                                               X[examples_i, feature_idx] <= percentile2))[0]
                    #     iqr1 = iqr(X[examples_i, feature_idx], rng=(iqr_rng1, self.iqr_range[1]), scale=1.0)
                    #     mean1 = np.mean(X[examples_i[picked_examples], feature_idx])
                    # if np.sign(mean2) < np.sign(iqr2):
                    #     iqr_rng1 = np.mean(self.iqr_range).astype(int)
                    #     percentile1 = np.percentile(X[examples_j, feature_idx], iqr_rng1)
                    #     percentile2 = np.percentile(X[examples_j, feature_idx], self.iqr_range[1])
                    #     picked_examples = np.where(np.logical_and(X[examples_j, feature_idx] >= percentile1,
                    #                                               X[examples_j, feature_idx] <= percentile2))[0]
                    #     iqr2 = iqr(X[examples_j, feature_idx], rng=(iqr_rng1, self.iqr_range[1]), scale=1.0)
                    #     mean2 = np.mean(X[examples_j[picked_examples], feature_idx])
                    # #############################
                    iqrs.append(iqr1 - iqr2)
                    means.append(mean1 - mean2)
                    ttests.append(statistic)

            # check if negative to separate classes for later
            if max(iqrs) <= 0:
                classes.append(0)
            else:
                classes.append(1)

            # append the top variance
            diff_iqrs.append(max(np.abs(iqrs)))
            diff_means.append(max(np.abs(means)))
            ttest_statistics.append(max(np.abs(ttests)))

        results = pd.concat([pd.DataFrame(diff_iqrs)], axis=1)
        results.columns = ['iqr']
        results['median_diff'] = diff_means
        results['ttest'] = ttest_statistics
        results['score'] = np.array(diff_iqrs) + np.array(diff_means)
        results['class_diff'] = classes

        results = results.to_numpy()

        return results
