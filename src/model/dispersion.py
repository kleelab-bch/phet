'''
Dispersion: XXX.

1. XXX.
'''

import numpy as np
import statsmodels.api as sm


class Dispersion:
    def __init__(self, score_function: str = "tanh", direction: str = "both",
                 calculate_pval: bool = True, num_iterations: int = 10000):
        self.score_function = score_function
        self.direction = direction  # up, down, both
        self.calculate_pval = calculate_pval
        self.num_iterations = num_iterations

    def fit_predict(self, X, y, normal_class: int = 0, test_class: int = 1):
        # Sanity checking
        if np.unique(y).shape[0] != 2:
            temp = "Only two valid groups are allowed!"
            raise Exception(temp)
        if test_class not in np.unique(y):
            temp = "Please provide a valid test group id!"
            raise Exception(temp)

        if num_features < 1:
            raise Exception("Features size should be greated than one!")

        num_examples, num_features = X.shape

        # Step 1: Estimate gene-wise dispersion
        gene_wise_dispersion = np.zeros((num_features, len(np.unique(y))))
        mu = np.zeros((num_features, len(np.unique(y))))
        variance = np.zeros((num_features, len(np.unique(y))))
        for class_idx in np.unique(y):
            temp_Xc = X[np.where(class_idx == y)[0]]
            for feature_idx in range(num_features):
                temp = np.ones_like(temp_Xc[:, feature_idx])
                res = sm.NegativeBinomial(endog=temp_Xc[:, feature_idx], exog=temp).fit(start_params=[1, 1])
                alpha = res.params[1]
                gene_wise_dispersion[feature_idx, class_idx] = alpha
                mu[feature_idx, class_idx] = np.exp(res.params[0])
                variance[feature_idx, class_idx] = mu[feature_idx, class_idx] + alpha * (
                        mu[feature_idx, class_idx] ** 2)
        del mu, variance, temp, temp_Xc, res

        # Step 1: Fit linear models for microarray and RNA seq data analysis
        coefficients = np.zeros((num_features, 2))
        std_unscaled = np.zeros((num_features, 2))
        sigma = np.zeros((num_features, 1))
        df_resid = list()
        for feature_idx in range(num_features):
            res = sm.GLM(endog=y, exog=sm.add_constant(X[:, feature_idx]),
                         family=sm.families.Gamma(link=sm.families.links.log())).fit(scale="X2")
            coefficients[feature_idx] = res.params
            sigma[feature_idx] = np.sqrt(np.sum(res.resid_deviance ** 2) / res.df_resid)
            std_unscaled[feature_idx] = np.sqrt(res.cov_params()[0])
            df_resid.append(res.df_resid)
        del res

        # Step 2: Fit curve to gene-wise dispersion estimates given normalized data
        fitted_values = np.zeros((num_features, len(np.unique(y))))
        for class_idx in np.unique(y):
            temp_Xc = X[np.where(class_idx == y)[0]]
            res = sm.GLM(endog=gene_wise_dispersion[:, class_idx],
                         exog=np.vstack((np.ones((num_features)), np.mean(temp_Xc, axis=0))).T,
                         family=sm.families.Gamma(link=sm.families.links.log())).fit(scale="X2")
            fitted_values[:, class_idx] = res.fittedvalues
        del res, temp_Xc

        # Step 3: Remove gene outliers given gene-wise dispersion estimates
        # look for the paper ORT to detect outliers
        keep_features = list()
        log_std = np.median(fitted_values, 0) / 1.4826
        for class_idx in np.unique(y):
            temp_Xc = X[np.where(class_idx == y)[0]]
            temp = np.log(fitted_values[:, class_idx]) + 1.5 * log_std[class_idx]
            temp = np.where(np.log(gene_wise_dispersion[:, class_idx]) <= temp)[0]
            keep_features.extend(temp)
        keep_features = np.unique(keep_features)
        X = X[:, keep_features]
        feature_names = np.array(feature_names)[keep_features].tolist()
        del X, fitted_values, gene_wise_dispersion, temp_Xc, keep_features, temp

        return X, feature_names


num_examples = 10
num_features = 5
X = np.random.normal(size=(num_examples, num_features))
y = np.random.randint(0, 2, num_examples)
