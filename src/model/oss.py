'''
Outlier sums for differential gene expression analysis.

1. Tibshirani, R. and Hastie, T., 2007. Outlier sums for 
differential gene expression analysis. Biostatistics, 8(1), 
pp.2-8.
'''

import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.stats import iqr


class OutlierSumStatistic:
    def __init__(self, q: float = 0.75, iqr_range: int = (25, 75), two_sided_test: bool = True,
                 direction: str = "both", calculate_pval: bool = True, num_iterations: int = 10000):
        self.q = q
        self.iqr_range = iqr_range
        self.two_sided_test = two_sided_test
        self.direction = direction  # up, down, both
        self.calculate_pval = calculate_pval
        self.num_iterations = num_iterations

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if case_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        num_features = X.shape[1]

        # Robustly standardize median
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.absolute(X - med), axis=0)
        X = (X - med) / mad

        # IQR estimation
        interquartile_range = iqr(X, axis=0, rng=self.iqr_range, scale=1.0)
        qr_pos = np.percentile(a=X, q=self.q, axis=0)
        qriqr_pos = qr_pos + interquartile_range
        qr_neg = np.percentile(a=X, q=1 - self.q, axis=0)
        qriqr_neg = qr_neg - interquartile_range

        # Include only test data
        control_X = X[np.where(y == control_class)[0]]
        case_X = X[np.where(y == case_class)[0]]

        # Find one-sided or two-sided stat
        os_pos = [case_X[np.where(case_X[:, idx] > qriqr_pos[idx])[0], idx].sum()
                  for idx in range(num_features)]
        if self.two_sided_test:
            os_neg = [case_X[np.where(case_X[:, idx] < qriqr_neg[idx])[0], idx].sum()
                      for idx in range(num_features)]
        else:
            os_neg = np.zeros_like(os_pos)
        results = np.c_[np.absolute(os_pos), np.absolute(os_neg)]
        results = np.max(results, axis=1)

        if self.calculate_pval:
            # Permutation based p-value calculation using approximate method
            pvals = np.zeros((num_features,))
            for feature_idx in range(num_features):
                if self.direction == "up":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean > y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                elif self.direction == "down":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean < y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                else:
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean != y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                pvals[feature_idx] += temp

            results = np.vstack((results, pvals)).T
        else:
            results = np.reshape(results, (results.shape[0], 1))

        return results
