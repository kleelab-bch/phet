import numpy as np
from scipy.stats import norm
class LSOSS:
    def __init__(self, q: float = 0.75):
        self.q = q

    def fit_predict(self, X, y, normal_class: int = 0, test_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if test_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        num_features = X.shape[1]
        examples1 = np.where(y == normal_class)[0]
        examples2 = np.where(y == test_class)[0]
        X1 = X[examples1]
        X2 = X[examples2]
        n = len(examples1)
        m = len(examples2)

        # Estimate the sum of squares for normal samples
        mean_normal = np.mean(X1, axis=0)
        SX1 = np.sum((X1 - mean_normal) ** 2, axis=0)

        # For each feature, the expression levels in test samples are sorted 
        # in descending order and then divided into two subsets
        results = np.zeros((num_features, ))
        for feature_idx in range(num_features):
            X_temp = np.sort(X2[:, feature_idx])[::-1]
            M = np.zeros((m - 1))
            for k in range(1, m - 1):
                S1 = X_temp[:k]
                S2 = X_temp[k:]
                # For the two subsets, the mean and sum of squares for each 
                # feature are calculated
                mean1 = np.mean(S1)
                mean2 = np.mean(S2)
                SS1 = np.sum((S1 - mean1) ** 2)
                SS2 = np.sum((S2 - mean2) ** 2)
                M[k] = SS1 + SS2
            k = np.argmin(M) + 1
            S1 = X_temp[:k]
            S2 = X_temp[k:]
            
            # For the two subsets, the mean and sum of squares for each 
            # feature are calculated
            mean1 = np.mean(S1)
            mean2 = np.mean(S2)
            SS1 = np.sum((S1 - mean1) ** 2)
            SS2 = np.sum((S2 - mean2) ** 2)

            # Estimate the pooled standard error
            SE = (SX1 + SS1 + SS2) / (n + m - 2)
            
            # The LSOSS statistic for declaring a feature with outlier differential 
            # expression in case samples is computed
            results[feature_idx] = k * ((SS1 - SX1) / SE)

        return results
