'''
TODO: 
1. Borrow ideas from MOST wrt Order Statistics
2. Link those methods with Outliers detection
'''

from itertools import combinations

import numpy as np
import pandas as pd
from prince import CA
from scipy.stats import iqr, zscore, ttest_ind
from scipy.stats import pearsonr, f_oneway, ks_2samp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from utility.utils import clustering

class UHeT:
    def __init__(self, normalize: str = None, q: float = 0.75, iqr_range: int = (25, 75),
                 calculate_pval: bool = True, subsamples: int = 100):
        self.normalize = normalize
        self.q = q
        self.iqr_range = iqr_range
        self.calculate_pval = calculate_pval
        self.subsamples = subsamples
        self.subsampling_size = 3
        self.significant_p = 0.15
        self.anova_test = False
        self.num_components = 10
        self.num_iterations = 50
        self.num_clusters = 10
        self.binary_clustering = True
        self.num_jobs = 4

    def __binary_partitioning(self, X):
        num_examples, num_features = X.shape
        M = np.zeros((num_features,))
        K = np.zeros((num_features, num_examples), dtype=np.int8)
        for feature_idx in range(num_features):
            temp = np.zeros((num_examples - 2))
            temp_X = np.sort(X[:, feature_idx])[::-1]
            order_list = np.argsort(X[:, feature_idx])[::-1]
            for k in range(1, num_examples - 1):
                S1 = temp_X[:k]
                S2 = temp_X[k:]
                if self.anova_test:
                    _, pvalue = f_oneway(S1, S2)
                    temp[k - 1] = pvalue
                else:
                    # For the two subsets, the mean and sum of squares for each
                    # feature are calculated
                    mean1 = np.mean(S1)
                    mean2 = np.mean(S2)
                    SS1 = np.sum((S1 - mean1) ** 2)
                    SS2 = np.sum((S2 - mean2) ** 2)
                    temp[k - 1] = SS1 + SS2
            k = np.argmin(temp) + 1
            M[feature_idx] = np.min(temp)
            K[feature_idx, order_list[k:]] = 1
        k = np.argmin(M)
        y = K[k]
        if self.binary_clustering:
            y = clustering(X=K.T, cluster_type="agglomerative", affinity="euclidean", num_neighbors=5,
                           num_clusters=2, num_jobs=2, predict=True)
        return y

    def fit_predict(self, X, y=None, control_class: int = 0, case_class: int = 1):
        # Extract properties
        num_examples, num_features = X.shape

        # Check if classes information is not provided (unsupervised analysis)
        if y is not None:
            if np.unique(y).shape[0] != 2:
                temp = "Only two valid groups are allowed!"
                raise Exception(temp)
            if control_class == case_class:
                temp = "Please provide two distinct groups ids!"
                raise Exception(temp)
            if control_class not in np.unique(y) or case_class not in np.unique(y):
                temp = "Please provide valid control/case group ids!"
                raise Exception(temp)
            num_classes = len(np.unique(y))
            control_examples = np.where(y == control_class)[0]
            case_examples = np.where(y == case_class)[0]
            n = len(control_examples)
            m = len(case_examples)
        else:
            # If there is no class information the algorithm will iteratively group
            # samples into two classes based on minimum within class mean differences
            # For each feature, the expression levels in all samples are sorted in
            # descending order and then divided into two subsets
            y = self.__binary_partitioning(X=X)
            num_classes = len(np.unique(y))
            control_examples = np.where(y == control_class)[0]
            case_examples = np.where(y == case_class)[0]
            n = len(control_examples)
            m = len(case_examples)

        if self.normalize == "robust":
            # Robustly estimate median by classes
            med = list()
            for class_idx in range(num_classes):
                if num_classes > 1:
                    class_idx = np.where(y == class_idx)[0]
                else:
                    class_idx = range(num_examples)
                example_med = np.median(X[class_idx], axis=0)
                temp = np.absolute(X[class_idx] - example_med)
                med.append(temp)
            med = np.median(np.concatenate(med), axis=0)
            X = X / med
            del class_idx, example_med, temp, med
        elif self.normalize == "zscore":
            X = zscore(X, axis=0)

        # Step 1: Recurrent-sampling differential analysis to select and rank
        # significant features
        if self.subsampling_size is not None:
            if num_classes == 2:
                temp = np.sqrt(np.min([n, m])).astype(int)
            else:
                temp = np.sqrt(num_examples).astype(int)
            self.subsampling_size = temp
        # Define frequency A and raw p-value P matrices
        A = np.ones((num_features, num_examples))
        new_P = np.zeros((num_features, self.subsamples))
        if num_classes == 2:
            A = np.zeros((2 * num_features, num_examples))
        A = A + 1
        # TODO: find a way to cluster samples into several groups and then
        #  iteratively partition samples (hierarchical partition)
        for sample_idx in range(self.subsamples):
            if num_classes == 2:
                subset = np.random.choice(a=control_examples, size=self.subsampling_size,
                                          replace=False)
                subset = np.concatenate([subset, np.random.choice(a=case_examples,
                                                                  size=self.subsampling_size,
                                                                  replace=False)])
            else:
                subset = np.random.choice(a=num_examples, size=self.subsampling_size,
                                          replace=False)
            iqrs = []
            medians = []
            ttests = []
            what_class = []
            for feature_idx in range(num_features):
                temp_iqrs = []
                temp_medians = []
                temp_ttests = []
                for i, j in combinations(range(num_classes), 2):
                    examples_i = np.where(y[subset] == i)[0]
                    examples_i = subset[examples_i]
                    temp_iqr = iqr(X[examples_i, feature_idx], rng=self.iqr_range, scale=1.0)
                    temp_med = np.median(X[examples_i, feature_idx])
                    # Make transposed matrix with shape (feat per class, observation per class)
                    # find mean and iqr difference between genes
                    if num_classes == 2:
                        examples_j = np.where(y[subset] == j)[0]
                        examples_j = subset[examples_j]
                        temp_iqr = temp_iqr - iqr(X[examples_j, feature_idx], rng=self.iqr_range, scale=1.0)
                        temp_med = temp_med - np.median(X[examples_j, feature_idx])
                    temp_iqrs.append(temp_iqr)
                    temp_medians.append(temp_med)
                    ttest, pvalue = ttest_ind(X[examples_i, feature_idx], X[examples_j, feature_idx])
                    if pvalue <= self.significant_p:
                        if num_classes == 2:
                            if ttest > 0:
                                A[feature_idx, examples_i] += 1
                            else:
                                A[feature_idx + num_features, examples_j] += 1
                        else:
                            A[feature_idx, examples_i] += 1
                        new_P[feature_idx, sample_idx] = pvalue
                    temp_ttests.append(ttest)
                # check if negative to separate classes for later
                if max(temp_iqrs) <= 0:
                    what_class.append(0)
                else:
                    what_class.append(1)
                # append the top variance
                iqrs.append(max(np.abs(temp_iqrs)))
                medians.append(max(np.abs(temp_medians)))
                ttests.append(max(np.abs(temp_ttests)))
        # Apply Fisher's method for combined probability 
        X_f = -2 * np.log(new_P)
        X_f[X_f == np.inf] = 0
        X_f = np.sum(X_f, axis=1)
        # # Calculate p-values from the Chi-square distribution with 2*self.subsets degrees of freedom
        # new_P = 1 - chi2.cdf(x = X_f, df = 2 * self.subsamples) 
        # # Adjusted p-values by BH method
        # _, new_P = fdrcorrection(pvals=new_P, alpha=0.05, is_sorted=False)
        # Standardize X_f
        X_f = (X_f - 2 * self.subsamples) / np.sqrt(4 * self.subsamples)

        # Step 2: Correspondence Analysis using frequency matrices
        ca = CA(n_components=self.num_components, n_iter=self.num_iterations, benzecri=False)
        ca.fit(X=A)

        # Step 3: Mapping the CA data for features and samples in a multidimensional space 
        P = cosine_similarity(ca.U_, ca.V_.T)
        # Estimate gene-wise dispersion
        D = np.zeros((num_features, num_examples))
        if num_classes == 2:
            D = np.zeros((2 * num_features, num_examples))
        for class_idx in np.unique(y):
            examples = np.where(class_idx == y)[0]
            temp = X[examples] - np.mean(X[examples], axis=0)
            if num_classes == 2:
                D[class_idx * num_features: (class_idx * num_features) + num_features, examples] = temp.T
            else:
                D[:, examples] = temp.T
        del examples, temp
        # Compute the heterogeneity statistic of each profile
        H = np.multiply(D, P) - np.mean(np.multiply(D, P), axis=1)[:, None]

        # Step 4: Finding all classes and groups in the sample set
        C = np.zeros((num_examples, num_examples))
        for i in range(num_examples):
            for j in range(i + 1, num_examples):
                C[i, j] = pearsonr(x=H[:, i], y=H[:, j])[0]
        C = C + C.T
        # Find an optimal number of k clusters-subclasses
        subclusters = self.num_clusters
        temp_scores = np.zeros((self.num_clusters - 2))
        for cluster_size in range(2, subclusters):
            model = AgglomerativeClustering(n_clusters=cluster_size)
            model.fit(X=C)
            temp = silhouette_score(X=C, labels=model.labels_)
            temp_scores[cluster_size - 2] = temp
        subclusters = np.argmax(temp_scores) + 2
        subclusters = clustering(X=C, cluster_type="agglomerative", affinity="euclidean", num_neighbors=5,
                           num_clusters=subclusters, num_jobs=self.num_jobs, predict=True)

        # Step 5: Identification of 4 feature profiles
        O = np.zeros((num_features, 4))
        for feature_idx in range(num_features):
            temp_lst = list()
            for i, j in combinations(range(num_classes), 2):
                temp = np.where(y == i)[0]
                temp = X[temp, feature_idx]
                if num_classes == 2:
                    examples_j = np.where(y == j)[0]
                    examples_j = X[examples_j, feature_idx]
                    pvalue = ks_2samp(temp, examples_j)[1]
                    temp_lst.append(pvalue)
            # Complete change
            if 0.2 > np.mean(temp_lst):
                O[feature_idx, 0] = 1
            # Majority change
            elif 0.4 > np.mean(temp_lst) >= 0.2:
                O[feature_idx, 1] = 1
            # Minority change
            elif 0.8 > np.mean(temp_lst) >= 0.4:
                O[feature_idx, 2] = 1
            # Mixed change
            else:
                O[feature_idx, 3] = 1

        # Step 6: Feature ranking based on combined parameters
        # Three main parameters are calculated: (i) standard Chi-Square value X_f, which
        # highlights the most significant and constant differential changes among samples;
        # (ii) h-statistic range per feature, which indicates how discriminant each feature
        # is, given the sample subclasses; and (iii) both of O and standard deviation of raw
        # omic signal in each differential feature, assessing the variability along samples
        # to allow finding the most stable features that will be considered the best markers
        # for the classes or subclasses found
        # X_f, H, O

        results = pd.concat([pd.DataFrame(iqrs)], axis=1)
        results.columns = ['iqr']
        results['median_diff'] = medians
        results['ttest'] = ttests
        results['score'] = np.array(medians) + np.array(iqrs)
        results['class_diff'] = what_class
        results = results.to_numpy()
        return results, H, O, subclusters


num_examples = 50
num_features = 200
control_class = 0
case_class = 1
y = np.random.randint(0, 2, num_examples)
X = np.zeros((num_examples, num_features))
temp = np.where(y == 0)[0]
X[temp] = np.random.normal(size=(len(temp), num_features))
temp = np.where(y == 1)[0]
X[temp] = np.random.normal(loc=2, scale=5, size=(len(temp), num_features))
model = UHeT(normalize="robust")
model.fit_predict(X=X, y=None, control_class=control_class,
                  case_class=case_class)
