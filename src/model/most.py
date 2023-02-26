'''
MOST: detecting cancer differential gene expression.

1. Lian, H., 2008. MOST: detecting cancer differential gene 
expression. Biostatistics, 9(3), pp.411-418.
'''

import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.stats import norm


class MOST:
    def __init__(self, k: int = None, direction: str = "both", permutation_test: bool = False,
                 num_rounds: int = 10000):
        self.k = k
        self.direction = direction  # up, down, both
        self.permutation_test = permutation_test
        self.num_rounds = num_rounds

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if case_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        num_features = X.shape[1]

        # Robustly estimate median by classes
        control_examples = np.where(y == control_class)[0]
        control_med = np.median(X[control_examples], axis=0)
        case_examples = np.where(y == case_class)[0]
        case_med = np.median(X[case_examples], axis=0)
        control_X = np.absolute(X[control_examples] - control_med)
        case_X = np.absolute(X[case_examples] - case_med)
        med = np.concatenate((control_X, case_X))
        med = np.median(med, axis=0) * 1.4826
        del control_X, case_X, case_med

        # Compute MOST test using test examples
        X = X[case_examples]
        m = len(case_examples)
        M = np.zeros((num_features, m - 2))
        if self.k != None:
            m = self.k
            M = np.zeros((num_features, m))

        for sample_idx in range(2, m):
            for feature_idx in range(num_features):
                loc, scale = norm.fit(X[:sample_idx, feature_idx])
                if scale == 0:
                    scale = 1
                X_temp = norm.cdf(X[:sample_idx, feature_idx], loc=loc, scale=scale)
                loc, scale = norm.fit(X_temp)
                if scale == 0:
                    scale = 1
                temp_idx = np.argsort(X_temp * -1)
                M[feature_idx, sample_idx - 2] = np.sum(X[temp_idx, feature_idx] - control_med[feature_idx])
                M[feature_idx, sample_idx - 2] /= med[feature_idx]
                M[feature_idx, sample_idx - 2] -= loc
                M[feature_idx, sample_idx - 2] /= scale
        np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.max(M, axis=1)

        if self.permutation_test:
            # Permutation based p-value calculation using approximate method
            pvals = np.zeros((num_features,))
            for feature_idx in range(num_features):
                if self.direction == "up":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean > y_mean", method="approximate",
                                            num_rounds=self.num_rounds)
                elif self.direction == "down":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean < y_mean", method="approximate",
                                            num_rounds=self.num_rounds)
                else:
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean != y_mean", method="approximate",
                                            num_rounds=self.num_rounds)
                pvals[feature_idx] += temp

            results = np.vstack((results, pvals)).T
        else:
            results = np.reshape(results, (results.shape[0], 1))

        return results
