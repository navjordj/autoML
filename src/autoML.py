import warnings
import datetime

import pandas as pd
import numpy as np
import numpy as numpy

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

from sklearn.externals import joblib

import xgboost


def warn(*args, **kwargs):
    pass

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

warnings.warn = warn


class autoML:
    def __init__(self, dataset, target, delimiter=None, drop_index=None):
        self.dataset = dataset
        self.data = self.read_dataset(self.dataset, delimiter)
        self.target = target
        if drop_index != None:
            self.drop_index = drop_index

        self.processed = False
        self.labels = None

        self.results = None
        self.algorithms = {
            'model': [LogisticRegression, SGDClassifier, SVC, xgboost.XGBClassifier],
            'hyperparameters': [{"C": np.logspace(-3, 3, 7),
                                 "penalty": ["l1", "l2"],
                                 "fit_intercept": [True, False],
                                 "solver": ['liblinear', 'saga']},
                                {
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'n_iter': [1000],
                'loss': ['log'],
                'penalty': ['l2'],
                'n_jobs': [-1]}, {
    'max_depth': [2], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
    'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],
    'n_estimators': [1000], #[1000,2000,3000]
    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]
}, {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}],
            'best_params': [],
            'best_acc': [],
        }

        self.best_model = None

    def process_data(self):
        """
        Drop index column if specified by user
        Drops columns with missing values
        """
        self.data = self.data.drop(self.drop_index, axis=1)
        self.data = self.data.dropna(axis='columns')
        self.label_target()

        self.processed = True

        self.bestacc = 0

    def label_target(self):
        enc = LabelEncoder()
        enc.fit(self.data[self.target])
        self.data[self.target] = enc.transform(self.data[self.target])
        self.labels = enc.classes_

    def remove_outliers(self):
        pass

    def find_best_model(self):

        X = self.data.drop([self.target], axis=1)
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.results = []
        for idx, algorithm in enumerate(tqdm(self.algorithms["model"])):
            algorithm_cv = GridSearchCV(
                algorithm(), self.algorithms["hyperparameters"][idx])
            algorithm_cv.fit(X_train, y_train)
            y_pred = algorithm_cv.predict(X_test)
            self.algorithms["best_params"].append(algorithm_cv.best_params_)
            self.algorithms["best_acc"].append(algorithm_cv.best_score_)
            if (algorithm_cv.best_score_ > self.bestacc):
                self.best = algorithm_cv.best_score_
                self.best_model = algorithm_cv.best_estimator_

    def save_model(self):
        model_file = f'model{datetime.date.today()}.pkl'
        joblib.dump(self.best_model, model_file)

    def read_dataset(self, dataset, delimiter):
        return pd.read_csv(dataset, delimiter=delimiter)

    def show_data(self):
        print(self.data.head())

    def __str__(self):
        return f'{self.dataset} \nprocessed = {self.processed}'


if __name__ == "__main__":
    data = autoML('../data.csv', drop_index='id', target="diagnosis")
    data.process_data()
    data.show_data()
    data.find_best_model()
    data.save_model()
    print(data.best_model)