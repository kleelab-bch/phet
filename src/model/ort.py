import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.stats import iqr


class OutlierRobustTstatistic:
    def __init__(self, q: float = 75, iqr_range: float = (25, 75), direction: str = "both",
                 permutation_test: bool = False, num_rounds: int = 10000):
        self.q = q
        self.iqr_range = iqr_range
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
        med = np.median(med, axis=0)
        del control_X, case_X

        # IQR estimation
        interquartile_range = iqr(X[control_examples], axis=0, rng=self.iqr_range, scale=1.0)
        qr = np.percentile(a=X[control_examples], q=self.q, axis=0)
        qriqr = qr + interquartile_range

        # Get samples indices
        U = [np.where(X[case_examples, feature_idx] > qriqr[feature_idx])[0]
             for feature_idx in range(num_features)]
        X = X[case_examples]

        # Compute ORT test
        results = list()
        for feature_idx in range(num_features):
            temp = np.sum(X[U[feature_idx], feature_idx] -
                          control_med[feature_idx]) / med[feature_idx]
            results.append(temp)
        results = np.array(results)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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
