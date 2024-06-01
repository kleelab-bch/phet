'''
MOST: detecting cancer differential gene expression.
Reproduced from: https://github.com/wyp1125/LSOSS

1. Lian, H., 2008. MOST: detecting cancer differential gene 
expression. Biostatistics, 9(3), pp.411-418.
'''

import numpy as np


class MOST:
    def __init__(self, direction: str = "up"):
        self.direction = direction

    def generate_order(self, num_samples, num_randoms=1000):
        sample = np.random.standard_normal(size=(num_samples, num_randoms))
        ordered = np.sort(sample, axis=0)[::-1]
        cumsumordered = np.cumsum(ordered, axis=0)
        mu = np.mean(cumsumordered, axis=1)
        cumsumordered = cumsumordered - mu[:, None]
        sigma = np.std(cumsumordered, axis=1)
        return mu, sigma

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
        med = (np.median(med, axis=0) + 0.01) * 1.4826
        del control_X, case_X

        # Compute MOST test using case examples
        case_X = X[case_examples]
        num_case = len(case_examples)
        mu, sigma = self.generate_order(num_samples=num_case)
        M = np.zeros((num_features, num_case))
        for order_idx in range(num_case):
            M[:, order_idx] = np.mean(case_X[:order_idx + 1], axis=0) - control_med
            M[:, order_idx] /= med
            M[:, order_idx] -= mu[order_idx]
            M[:, order_idx] /= sigma[order_idx]
        del num_case, mu, sigma
        np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.direction == "both" or self.direction == "down":
            # Compute MOST test using control examples
            control_X = X[control_examples]
            num_control = len(control_examples)
            mu, sigma = self.generate_order(num_samples=num_control)
            N = np.zeros((num_features, num_control))
            for order_idx in range(num_control):
                N[:, order_idx] = np.mean(control_X[:order_idx + 1], axis=0) - case_med
                N[:, order_idx] /= med
                N[:, order_idx] -= mu[order_idx]
                N[:, order_idx] /= sigma[order_idx]
            del control_X, num_control, sigma, mu
            np.nan_to_num(N, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.direction == "both" or self.direction == "down":
            results = np.max(np.c_[M, N], axis=1)
        else:
            results = np.max(M, axis=1)
        results = np.reshape(results, (results.shape[0], 1))
        return results
