import numpy as np
from scipy.stats import iqr


class OutlierStatistic:
    def __init__(self, q: float = 0.75, iqr_range: int = (25, 75), two_sided_test: bool = True):
        self.q = q
        self.iqr_range = iqr_range
        self.two_sided_test = two_sided_test

    def fit_predict(self, X, y, test_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if test_class not in np.unique(y):
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
        X = X[np.where(y == test_class)[0]]

        # Find one-sided or two-sided stat
        os_pos = [X[np.where(X[:, idx] > qriqr_pos[idx])[0], idx].sum()
                  for idx in range(num_features)]
        if self.two_sided_test:
            os_neg = [X[np.where(X[:, idx] < qriqr_neg[idx])[0], idx].sum()
                      for idx in range(num_features)]
        else:
            os_neg = np.zeros_like(os_pos)
        results = np.max(np.c_[np.absolute(os_pos), np.absolute(os_neg)], axis=1)

        return results
