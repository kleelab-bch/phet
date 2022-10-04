'''
LSOSS: detection of cancer outlier differential gene
expression.

1. Wang, Y. and Rekaya, R., 2010. LSOSS: detection of cancer 
outlier differential gene expression. Biomarker insights, 5, 
pp.BMI-S5175.
'''

import numpy as np
from mlxtend.evaluate import permutation_test


class LSOSS:
    def __init__(self, k: int = None, direction: str = "both", calculate_pval: bool = False,
                 num_iterations: int = 10000):
        self.k = k
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
        control_examples = np.where(y == control_class)[0]
        case_examples = np.where(y == case_class)[0]
        control_X = X[control_examples]
        case_X = X[case_examples]
        n = len(control_examples)
        m = len(case_examples)

        # Estimate the sum of squares for normal samples
        mean_control = np.mean(control_X, axis=0)
        SS_control = np.sum((control_X - mean_control) ** 2, axis=0)

        # For each feature, the expression levels in test samples are sorted
        # in descending order and then divided into two subsets
        results = np.zeros((num_features,))
        for feature_idx in range(num_features):
            temp_X = np.sort(case_X[:, feature_idx])[::-1]
            M = np.zeros((m - 2))
            for k in range(1, m - 1):
                S1 = temp_X[:k]
                S2 = temp_X[k:]
                # For the two subsets, the mean and sum of squares for each
                # feature are calculated
                mean1 = np.mean(S1)
                mean2 = np.mean(S2)
                SS1 = np.sum((S1 - mean1) ** 2)
                SS2 = np.sum((S2 - mean2) ** 2)
                M[k - 1] = SS1 + SS2
            k = np.argmin(M) + 1
            S1 = temp_X[:k]
            S2 = temp_X[k:]

            # For the two subsets, the mean and sum of squares for each
            # feature are calculated
            mean1 = np.mean(S1)
            mean2 = np.mean(S2)
            SS1 = np.sum((S1 - mean1) ** 2)
            SS2 = np.sum((S2 - mean2) ** 2)

            # Estimate the pooled standard error
            SE = (SS_control[feature_idx] + SS1 + SS2) / (n + m - 2)

            # The LSOSS statistic for declaring a feature with outlier differential
            # expression in case samples is computed
            results[feature_idx] = k * ((mean1 - mean_control[feature_idx]) / SE)

            if self.direction == "both":
                results[feature_idx] += (m - k) * ((mean2 - mean_control[feature_idx]) / SE)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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
