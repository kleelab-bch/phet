'''
COPA—cancer outlier profile analysis.

1. MacDonald, J.W. and Ghosh, D., 2006. COPA—cancer outlier 
profile analysis. Bioinformatics, 22(23), pp.2950-2951.
'''

import numpy as np


class COPA:
    def __init__(self, q: float = 75):
        self.q = q

    def fit_predict(self, X, y, control_class: int = 0, case_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if case_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        # Compute column-wise the median of expression values
        # and the median absolute deviation of expression values
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.absolute(X - med), axis=0)

        # Include only test data
        case_X = X[np.where(y == case_class)[0]]

        # Calculate statistics
        results = (np.percentile(a=case_X, q=self.q, axis=0) - med) / mad
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results[results < 0] = 0
        results += 0.05
        results = np.reshape(results, (results.shape[0], 1))
        return results
