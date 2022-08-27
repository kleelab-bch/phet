import numpy as np
from scipy.stats import iqr


class OutlierRobustStatistic:
    def __init__(self, q: float = 0.75, iqr_range: int = (25, 75)):
        self.q = q
        self.iqr_range = iqr_range

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
        med = np.median(med, axis=0)
        del X1, X2

        # IQR estimation
        interquartile_range = iqr(X[examples1], axis=0, rng=self.iqr_range, scale=1.0)
        qr = np.percentile(a=X[examples1], q=self.q, axis=0)
        qriqr = qr + interquartile_range

        # Get samples indices
        U = [np.where(X[examples2, feature_idx] > qriqr[feature_idx])[0]
             for feature_idx in range(num_features)]
        X = X[examples2]

        # Compute ORT test
        results = list()
        for feature_idx in range(num_features):
            temp = np.sum(X[U[feature_idx], feature_idx] - med1[feature_idx]) / med[feature_idx]
            results.append(temp)
        results = np.array(results)
        return results
