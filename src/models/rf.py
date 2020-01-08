'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2019/7/22 
    Python Version: 3.6
'''

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit


class RandomForestModel(object):
    def __init__(self, n_estimators, max_depth):
        self.grid_parameters = {'n_estimators': n_estimators,
                                'max_depth': max_depth}

    def build_model(self):
        pass

    def load(self, fname):
        with open(fname, 'rb') as ifile:
            self.model = pickle.load(ifile)

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

        train_len = len(y_train)
        valid_len = len(y_valid)
        valid_index = [i for i in range(train_len, train_len + valid_len)]
        test_fold = [-1 if i not in valid_index else 0 for i in range(0, train_len + valid_len)]
        ps = PredefinedSplit(test_fold=test_fold)

        rfc = RandomForestClassifier(random_state=0, class_weight='balanced')

        self.model = GridSearchCV(rfc, self.grid_parameters, cv=ps,
                                  scoring='f1', verbose=0, n_jobs=2)

        self.model.fit(x_train_valid, y_train_valid)

    def predict(self, x_test):
        x_test = self.arr_concate(x_test)
        pred_class = self.model.predict(x_test)
        pred_score = self.model.predict_proba(x_test)[:, 1]
        return pred_class, pred_score

    def save(self, fname):
        with open(fname, 'wb') as ofile:
            pickle.dump(self.model, ofile, pickle.HIGHEST_PROTOCOL)


