import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


def classifiers_results(X, y, features_name: list, num_features: int, standardize: bool = True, num_jobs: int = 2,
                        save_path=""):
    # Sanity checking
    if X.shape[0] != y.shape[0]:
        temp = "The number of samples must be same for both X and y."
        raise Exception(temp)

    # Standardize if needed
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X=X)
        X = scaler.transform(X=X)

    classifiers_dict = {
        "DecisionTree": (DecisionTreeClassifier(),
                         {'max_depth': np.arange(2, 10),
                          'min_samples_leaf': np.arange(2, 5)}),
        "RandomForest": (RandomForestClassifier(n_jobs=num_jobs),
                         {'max_depth': np.arange(2, 10), 'min_samples_leaf': np.arange(2, 5),
                          'n_estimators': np.arange(5, 30, 5)}),
        "AdaBoost": (AdaBoostClassifier(),
                     {'n_estimators': np.arange(5, 30, 5)}),
        "LogisticRegression": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, n_jobs=num_jobs),
                               {'C': np.logspace(-2, 4, 15), 'l1_ratio': np.linspace(0, 1, 10)}),
        "SVM": (SVC(probability=True, random_state=12345),
                {'kernel': ('linear', 'rbf'), 'C': np.logspace(-2, 4, 15),
                 'degree': np.arange(1, 11),
                 'gamma': np.logspace(-9, 3, 13)})
    }

    current_progress = 0
    total_progress = len(classifiers_dict)
    best_score = 0
    ranked_features = list()
    estimators = list()
    for cls_name, item in classifiers_dict.items():
        current_progress += 1
        desc = '\t>> Grid search cross validation and feature selection for {0} ({1:.2f}%)...'.format(cls_name,
                                                                                                      current_progress / total_progress * 100)
        if current_progress == total_progress:
            print(desc)
        else:
            print(desc, end="\r")

        estimator, parameters = item
        cls = GridSearchCV(estimator, parameters)
        cls.fit(X=X, y=y)
        score = cls.best_score_
        if score > best_score:
            best_score = score

        # Select features based on RFE
        cls = cls.best_estimator_
        estimators.append((cls_name, cls))
        selector = RFE(cls, n_features_to_select=num_features)
        selector = selector.fit(X, y)
        ranked_features.append(np.array(features_name)[np.argsort(selector.ranking_)].tolist())
        # feature_name = pd.Series(features_name)
        # feature_name.index = selector.ranking_
        # df = pd.DataFrame(list(feature_name.sort_index().reset_index(drop=True)))

    df = pd.DataFrame(ranked_features)
    df.columns = list(classifiers_dict.keys())
    df.to_csv(path_or_buf=os.path.join(save_path, "ranked_features.tsv"), sep='\t')

    return df, estimators
