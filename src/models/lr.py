'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/30 
    Python Version: 3.6
'''

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit

import numpy as np
from sklearn.externals import joblib


class LRModel(object):
    def __init__(self, tuned_parameters):
        self.grid_parameters = tuned_parameters

    def build_model(self):
        pass

    def load(self, fname):
        with open(fname, 'rb') as ifile:
            # self.model = pickle.load(ifile)
            self.model = joblib.load(ifile)
        # print model detail
        print(self.model.get_params())

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

        self.model = LogisticRegressionCV(
            Cs=self.grid_parameters['Cs']
            # , penalty='l2'
            , penalty='elasticnet'
            , scoring='f1'
            , solver='saga'
            , cv=ps
            , random_state=0
            , max_iter=10000
            , class_weight="balanced"
            # ,fit_intercept=True
            , fit_intercept=False
            , tol=10
            , refit=True
            , l1_ratios=self.grid_parameters['l1_ratios']
        )

        self.model.fit(x_train_valid, y_train_valid)

    def predict(self, x_test):
        x_test = self.arr_concate(x_test)
        pred_class = self.model.predict(x_test)
        pred_score = self.model.predict_proba(x_test)[:, 1]
        return pred_class, pred_score

    def save(self, fname):
        with open(fname, 'wb') as ofile:
            joblib.dump(self.model, ofile, compress=1)

