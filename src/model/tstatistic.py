import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.stats import ttest_ind


class StudentTTest:
    def __init__(self, direction: str = "both", permutation_test: bool = False,
                 num_rounds: int = 10000):
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
        control_examples = np.where(y == control_class)[0]
        control_X = np.absolute(X[control_examples])
        case_examples = np.where(y == case_class)[0]
        case_X = np.absolute(X[case_examples])
        results = np.zeros((num_features))
        for feature_idx in range(num_features):
            if self.direction == "up":
                alternative = 'greater'
            elif self.direction == "down":
                alternative = "less"
            else:
                alternative = "two-sided"
            statistic, _ = ttest_ind(control_X[:, feature_idx],
                                     case_X[:, feature_idx],
                                     alternative=alternative)
            results[feature_idx] = statistic

        # Calculate statistics
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
