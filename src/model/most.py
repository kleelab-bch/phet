import numpy as np
from scipy.stats import norm


class MOST:
    def __init__(self, k: int = None):
        self.k = k

    def fit_predict(self, X, y, normal_class: int = 0, test_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if test_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        num_features = X.shape[1]

        # Robustly estimate median by classes
        examples1 = np.where(y == normal_class)[0]
        med1 = np.median(X[examples1], axis=0)
        examples2 = np.where(y == test_class)[0]
        med2 = np.median(X[examples2], axis=0)
        X1 = np.absolute(X[examples1] - med1)
        X2 = np.absolute(X[examples2] - med2)
        med = np.concatenate((X1, X2))
        med = np.median(med, axis=0) * 1.4826
        del X1, X2, med2

        # Compute MOST test using test examples
        X = X[examples2]
        M = np.zeros_like(X).T
        m = len(examples2)
        if self.k != None:
            m = self.k
            M = np.zeros((num_features, m))

        for sample_idx in range(2, m):
            for feature_idx in range(num_features):
                loc, scale = norm.fit(X[:sample_idx, feature_idx])
                X_temp = norm.cdf(X[:sample_idx, feature_idx], loc=loc, scale=scale)
                loc, scale = norm.fit(X_temp)
                temp_idx = np.argsort(X_temp * -1)
                M[feature_idx, sample_idx] = np.sum(X[temp_idx, feature_idx] - med1[feature_idx])
                M[feature_idx, sample_idx] /= med[feature_idx]
                M[feature_idx, sample_idx] -= loc
                M[feature_idx, sample_idx] /= scale

        results = np.max(M, axis=1)

        return results
