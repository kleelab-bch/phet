'''
COPA—cancer outlier profile analysis.

1. MacDonald, J.W. and Ghosh, D., 2006. COPA—cancer outlier 
profile analysis. Bioinformatics, 22(23), pp.2950-2951.
'''

import numpy as np
from mlxtend.evaluate import permutation_test


class COPA:
    def __init__(self, q: float = 0.75, direction: str = "both", calculate_pval: bool = False,
                 num_iterations: int = 10000):
        self.q = q
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
        # Compute column-wise the median of expression values
        # and the median absolute deviation of expression values
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.absolute(X - med), axis=0)

        # Include only test data
        control_X = X[np.where(y == control_class)[0]]
        case_X = X[np.where(y == case_class)[0]]

        # Calculate statistics
        results = (np.percentile(a=case_X, q=100 * self.q, axis=0) - med) / mad
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
