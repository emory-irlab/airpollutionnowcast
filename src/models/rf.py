'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2019/7/22 
    Python Version: 3.6
'''

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from src.data.utils import year_column
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib


class RandomForestModel(object):
    def __init__(self, n_estimators, max_depth):
        self.grid_parameters = {'n_estimators': n_estimators,
                                'max_depth': max_depth}

    def build_model(self):
        pass

    def load(self, fname):
        with open(fname, 'rb') as ifile:
            # self.model = pickle.load(ifile)
            self.model = joblib.load(ifile)

    def arr_concate(self, input_arr):
        if isinstance(input_arr, tuple):
            input_arr = np.concatenate(input_arr, axis=2)
        input_arr = input_arr.reshape(len(input_arr), -1)
        return input_arr

    def fit(self, x_train, x_valid, y_train, y_valid):
        def arr_append(train_arr, valid_arr):
            return np.concatenate([train_arr, valid_arr], axis=0)

        x_train = self.arr_concate(x_train)
        x_valid = self.arr_concate(x_valid)
        x_train_valid = arr_append(x_train, x_valid)
        y_train_valid = arr_append(y_train, y_valid)
        y_train_valid = pd.DataFrame(data=y_train_valid, columns=['label', year_column])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf_generator = skf.split(y_train_valid, y_train_valid[year_column])

        rfc = RandomForestClassifier(random_state=0, class_weight='balanced')

        self.model = GridSearchCV(rfc, self.grid_parameters, cv=skf_generator,
                                  scoring='roc_auc_score', verbose=0, n_jobs=2)
        y_train_valid.drop([year_column], axis=1, inplace=True)
        y_train_valid = np.array(y_train_valid).ravel()

        self.model.fit(x_train_valid, y_train_valid)

    def predict(self, x_test):
        x_test = self.arr_concate(x_test)
        pred_class = self.model.predict(x_test)
        pred_score = self.model.predict_proba(x_test)[:, 1]
        return pred_class, pred_score

    def save(self, fname):
        with open(fname, 'wb') as ofile:
            joblib.dump(self.model.best_estimator_, ofile, compress=1)
            # pickle.dump(self.model, ofile, pickle.HIGHEST_PROTOCOL)


