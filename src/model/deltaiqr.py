'''
Unraveling cellular Heterogeneity by analyzing intra-cellular
variation.
'''

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import iqr
from scipy.stats import zscore


class DiffIQR:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75),
                 calculate_pval: bool = False, num_iterations: int = 10000):
        self.normalize = normalize
        self.q = q
        self.iqr_range = iqr_range
        self.calculate_pval = calculate_pval
        self.num_iterations = num_iterations

    def fit_predict(self, X, y):
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
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return results