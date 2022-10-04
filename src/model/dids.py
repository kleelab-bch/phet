'''
Identifying subgroup markers in heterogeneous populations.

1. de Ronde, J.J., Rigaill, G., Rottenberg, S., Rodenhuis, S. 
and Wessels, L.F., 2013. Identifying subgroup markers in 
heterogeneous populations. Nucleic acids research, 41(21), 
pp.e200-e200.
'''

import numpy as np

from mlxtend.evaluate import permutation_test


class DIDS:
    def __init__(self, score_function: str = "tanh", direction: str = "both",
                 calculate_pval: bool = False, num_iterations: int = 10000):
        self.score_function = score_function
        self.direction = direction  # up, down, both
        self.calculate_pval = calculate_pval
        self.num_iterations = num_iterations

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
        case_examples = np.where(y == case_class)[0]
        control_X = X[control_examples]
        case_X = X[case_examples]

        # Estimate the maximal/minimal gene expression value of genes among
        # the control samples
        if self.direction == "up":
            control_hat = np.max(control_X, axis=0)
            temp_X2 = case_X - control_hat
            temp_X2[temp_X2 < 0] = 0
        elif self.direction == "down":
            control_hat = np.min(control_X, axis=0)
            temp_X2 = control_hat - case_X
            temp_X2[temp_X2 < 0] = 0
        else:
            # Compute upregulated features
            control_hat = np.max(control_X, axis=0)
            temp = case_X - control_hat
            temp[temp < 0] = 0
            # Compute downregulated features
            control_hat = np.min(control_X, axis=0)
            temp_X2 = control_hat - case_X
            temp_X2[temp_X2 < 0] = 0
            # Add both expressed features
            temp_X2 = temp + temp_X2

        # Compute the DIDS score
        if self.score_function == "quad":
            temp_X2 = temp_X2 ** 2
        elif self.score_function == "sqrt":
            temp_X2 = np.sqrt(temp_X2)
        else:
            temp_X2 = 3 * temp_X2 - 3
            temp_X2 = 1 + np.tanh(temp_X2)
        results = np.sum(temp_X2, axis=0)
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if self.calculate_pval:
            # Permutation based p-value calculation using approximate method
            pvals = np.zeros((num_features,))
            for feature_idx in range(num_features):
                if self.direction == "up":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean > y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                elif self.direction == "down":
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean < y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                else:
                    temp = permutation_test(x=control_X[:, feature_idx], y=case_X[:, feature_idx],
                                            func="x_mean != y_mean", method="approximate",
                                            num_rounds=self.num_iterations)
                pvals[feature_idx] += temp

            results = np.vstack((results, pvals)).T
        else:
            results = np.reshape(results, (results.shape[0], 1))

        return results
