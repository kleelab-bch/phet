'''
MOST: detecting cancer differential gene expression.

1. Lian, H., 2008. MOST: detecting cancer differential gene 
expression. Biostatistics, 9(3), pp.411-418.
'''

import numpy as np


class MOST:
    def __init__(self, direction: str = "up"):
        self.direction = direction

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
        del control_X, case_X

        # Compute MOST test using test examples
        case_X = X[case_examples]
        m_case = len(case_examples)
        M = np.zeros((num_features, m_case - 2))
        for sample_idx in range(2, m_case):
            loc = np.mean(case_X[:sample_idx], axis=0)
            scale = np.std(case_X[:sample_idx], axis=0)
            scale[scale == 0] = 1
            for feature_idx in range(num_features):
                M[feature_idx, sample_idx - 2] = np.sum(case_X[:sample_idx, feature_idx] - control_med[feature_idx])
                M[feature_idx, sample_idx - 2] /= med[feature_idx]
                M[feature_idx, sample_idx - 2] -= loc[feature_idx]
                M[feature_idx, sample_idx - 2] /= scale[feature_idx]
        del m_case, loc, scale
        np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.direction == "both" or self.direction == "down":
            # Compute MOST test using control examples
            control_X = X[control_examples]
            n_control = len(control_examples)
            N = np.zeros((num_features, n_control - 2))
            for sample_idx in range(2, n_control):
                loc = np.mean(control_X[:sample_idx], axis=0)
                scale = np.std(control_X[:sample_idx], axis=0)
                scale[scale == 0] = 1
                for feature_idx in range(num_features):
                    N[feature_idx, sample_idx - 2] = np.sum(control_X[:sample_idx, feature_idx] - case_med[feature_idx])
                    N[feature_idx, sample_idx - 2] /= med[feature_idx]
                    N[feature_idx, sample_idx - 2] -= loc[feature_idx]
                    N[feature_idx, sample_idx - 2] /= scale[feature_idx]
            del control_X, n_control, scale, loc
            np.nan_to_num(N, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.direction == "both" or self.direction == "down":
            results = np.max(np.c_[M, N], axis=1)
        else:
            results = np.max(M, axis=1)
        results = np.reshape(results, (results.shape[0], 1))
        return results
