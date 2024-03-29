import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ztest

EPSILON = np.finfo(np.float32).eps


class StudentTTest:
    def __init__(self, use_statistics: bool = False, direction: str = "both",
                 perform_permutation: bool = False, adjust_pvalue: bool = True,
                 adjusted_alpha=0.01, num_rounds: int = 10000):
        self.use_statistics = use_statistics
        self.direction = direction  # up, down, both
        self.perform_permutation = perform_permutation
        self.adjust_pvalue = adjust_pvalue
        self.adjusted_alpha = adjusted_alpha
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
        control_X = X[control_examples]
        case_examples = np.where(y == case_class)[0]
        case_X = X[case_examples]
        permutations = None
        if self.perform_permutation:
            permutations = self.num_rounds
        results = np.zeros((num_features))
        for feature_idx in range(num_features):
            if self.direction == "up":
                alternative = 'greater'
            elif self.direction == "down":
                alternative = "less"
            else:
                alternative = "two-sided"
            statistic, pvalue = ttest_ind(control_X[:, feature_idx],
                                          case_X[:, feature_idx],
                                          permutations=permutations,
                                          alternative=alternative)
            results[feature_idx] = pvalue
            if self.use_statistics:
                results[feature_idx] = np.absolute(statistic)
        if not self.use_statistics:
            if self.adjust_pvalue:
                results = fdrcorrection(results, alpha=self.adjusted_alpha + EPSILON,
                                        method="indep")[1]
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results


class ZTest:
    def __init__(self, use_statistics: bool = False, direction: str = "both",
                 adjust_pvalue: bool = True, adjusted_alpha=0.01,
                 num_rounds: int = 10000):
        self.use_statistics = use_statistics
        self.direction = direction  # up, down, both
        self.adjust_pvalue = adjust_pvalue
        self.adjusted_alpha = adjusted_alpha
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
        control_X = X[control_examples]
        case_examples = np.where(y == case_class)[0]
        case_X = X[case_examples]
        results = np.zeros((num_features))
        for feature_idx in range(num_features):
            if self.direction == "up":
                alternative = 'greater'
            elif self.direction == "down":
                alternative = "less"
            else:
                alternative = "two-sided"
            statistic, pvalue = ztest(control_X[:, feature_idx],
                                      case_X[:, feature_idx],
                                      alternative=alternative)
            results[feature_idx] = pvalue
            if self.use_statistics:
                results[feature_idx] = np.absolute(statistic)
        if not self.use_statistics:
            if self.adjust_pvalue:
                results = fdrcorrection(results, alpha=self.adjusted_alpha + EPSILON,
                                        method="indep")[1]
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results


class WilcoxonRankSumTest:
    def __init__(self, use_statistics: bool = False, direction: str = "both",
                 adjust_pvalue: bool = True, adjusted_alpha=0.01):
        self.use_statistics = use_statistics
        self.direction = direction  # up, down, both
        self.adjust_pvalue = adjust_pvalue
        self.adjusted_alpha = adjusted_alpha

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
        control_X = X[control_examples]
        case_examples = np.where(y == case_class)[0]
        case_X = X[case_examples]
        results = np.zeros((num_features))
        for feature_idx in range(num_features):
            if self.direction == "up":
                alternative = 'greater'
            elif self.direction == "down":
                alternative = "less"
            else:
                alternative = "two-sided"
            statistic, pvalue = mannwhitneyu(control_X[:, feature_idx],
                                             case_X[:, feature_idx],
                                             alternative=alternative)
            results[feature_idx] = pvalue
            if self.use_statistics:
                results[feature_idx] = np.absolute(statistic)
        if not self.use_statistics:
            if self.adjust_pvalue:
                results = fdrcorrection(results, alpha=self.adjusted_alpha + EPSILON,
                                        method="indep")[1]
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results


class KolmogorovSmirnovTest:
    def __init__(self, use_statistics: bool = False, direction: str = "both",
                 adjust_pvalue: bool = True, adjusted_alpha=0.01):
        self.use_statistics = use_statistics
        self.direction = direction  # up, down, both
        self.adjust_pvalue = adjust_pvalue
        self.adjusted_alpha = adjusted_alpha

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
        control_X = X[control_examples]
        case_examples = np.where(y == case_class)[0]
        case_X = X[case_examples]
        results = np.zeros((num_features))
        for feature_idx in range(num_features):
            if self.direction == "up":
                alternative = 'greater'
            elif self.direction == "down":
                alternative = "less"
            else:
                alternative = "two-sided"
            statistic, pvalue = ks_2samp(control_X[:, feature_idx],
                                         case_X[:, feature_idx],
                                         alternative=alternative)
            results[feature_idx] = pvalue
            if self.use_statistics:
                results[feature_idx] = np.absolute(statistic)
        if not self.use_statistics:
            if self.adjust_pvalue:
                results = fdrcorrection(results, alpha=self.adjusted_alpha + EPSILON,
                                        method="indep")[1]
        np.nan_to_num(results, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        results = np.reshape(results, (results.shape[0], 1))
        return results
