import numpy as np


class COPA:
    def __init__(self, q: float = 0.75):
        self.q = q

    def fit_predict(self, X, y, test_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if test_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        # Compute column-wise the median of expression values
        # and the median absolute deviation of expression values
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.absolute(X - med), axis=0)

        # Include only test data
        X = X[np.where(y == test_class)[0]]

        # Calculate statistics
        results = (np.percentile(a=X, q=100 * self.q, axis=0) - med) / mad

        return results
