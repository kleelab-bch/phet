import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import iqr
from scipy.stats import zscore


class UHeT:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75)):
        self.normalize = normalize
        self.q = q
        self.iqr_range = iqr_range

    def fit_predict(self, X, y):
        """
        Hetero-Net Function

        Perform Deep Metric Learning with UMAP-based clustering to find subpopulations of classes

        Read more in the USER GUIDE

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples

        standardize : bool, default=True
            Standardizes data using zscore (NOTE: test more?)

        Attributes
        ----------
        features_ : list
            The features found by the algorithm. Ranked in order of importance

        score_ : list
            The score found by the algorithm for each feature


        References
        ----------
        NOTE: MINE


        Examples
        ----------
        NOTE: TODO
        """

        # Extract properties
        num_classes = len(np.unique(y))
        num_features = X.shape[1]

        if self.normalize == "robust":
            # Robustly estimate median by classes
            med = list()
            for i in range(num_classes):
                example_idx = np.where(y == i)[0]
                example_med = np.median(X[example_idx], axis=0)
                temp = np.absolute(X[example_idx] - example_med)
                med.append(temp)
            med = np.median(np.concatenate(med), axis=0)
            X = X / med
            del example_idx, example_med, temp, med
        elif self.normalize == "zscore":
            X = zscore(X, axis=0)

        # make transposed matrix with shape (feat per class, observation per class)
        # find mean and iqr difference between genes
        var_ls = []
        mean_ls = []
        ttest_ls = []
        what_class = []
        for p in range(num_features):
            temp_lst = []
            temp_mean_lst = []
            temp_ttest_lst = []
            for i in range(num_classes):
                examples_i = np.where(y == i)[0]
                for j in range(i + 1, num_classes):
                    examples_j = np.where(y == j)[0]
                    temp = iqr(X[examples_i, p], rng=self.iqr_range, scale=1.0)
                    temp = temp - iqr(X[examples_j, p], rng=self.iqr_range, scale=1.0)
                    temp_lst.append(temp)
                    temp = np.mean(X[examples_i, p])
                    temp = temp - np.mean(X[examples_j, p])
                    temp_mean_lst.append(temp)
                    temp = stats.ttest_ind(X[examples_i, p], X[examples_j, p])[0]
                    temp_ttest_lst.append(temp)

            # check if negative to seperate classes for later
            if max(temp_lst) <= 0:
                what_class.append(0)
            else:
                what_class.append(1)

            # append the top variance
            var_ls.append(max(np.abs(temp_lst)))
            mean_ls.append(max(np.abs(temp_mean_lst)))
            ttest_ls.append(max(np.abs(temp_ttest_lst)))

        results = pd.concat([pd.DataFrame(var_ls)], axis=1)
        results.columns = ['iqr']
        results['median_diff'] = mean_ls
        results['ttest'] = ttest_ls
        results['score'] = np.array(mean_ls) + np.array(var_ls)
        results['class_diff'] = what_class

        results = results.to_numpy()
        return results
