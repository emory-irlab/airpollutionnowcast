'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/7 
    Python Version: 3.6
'''

import sys
import os

sys.path.append('.')
from src.data import read_raw_data
from src.features.build_features import process_data, get_pol_value_series, lag_search_features, get_feature_array
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from src.models.rf import RandomForestModel
from src.models.composed_lstm import ComposedLSTM
import ast


RECORD_COLUMNS = ['accuracy', 'F1 score', 'true positives', 'false positives', 'true negatives',
                  'false negatives', 'AUC']


# read data and process to features
def process_features(train_data_path, seq_length, search_lag):
    train_data = read_raw_data(train_data_path)
    y_data, pol_val, trend_fea, phys_fea = process_data(train_data)
    processed_pol = get_pol_value_series(pol_val, seq_length)
    processed_trend = lag_search_features(trend_fea, search_lag)
    # fill NAs with 0 for phys_fea
    phys_fea.fillna(0, inplace=True)
    process_phys = np.array(phys_fea)
    return y_data, processed_pol, process_phys, processed_trend


# get random forest model
def get_rf_model(pars):
    # parameters for rf model
    n_estimators = ast.literal_eval(pars['train_model']['n_estimators'])
    max_depth = ast.literal_eval(pars['train_model']['max_depth'])
    model = RandomForestModel(n_estimators, max_depth)
    return model


def get_two_branch_feature(train_pol, train_phys, train_trend, seq_length):
    x_train_sensor, embedding_sensor = get_feature_array([train_pol, train_phys], seq_length)
    x_train_trend, embedding_trend = get_feature_array([train_trend], seq_length)

    x_train = (x_train_sensor, x_train_trend)
    embedding = (embedding_sensor, embedding_trend)
    return x_train, embedding


def get_lstm_model(pars, seq_length, embedding_dim):
    learning_rate = float(pars['train_model']['learning_rate'])
    batch_size = int(pars['train_model']['batch_size'])
    patience = int(pars['train_model']['patience'])
    log_dir = pars['train_model']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = ComposedLSTM(seq_length, embedding_dim, learning_rate, batch_size, patience, log_dir)
    return model


"""
Utils for predict_model.py
"""

# evaluation metrics
def result_stat(y_true, y_prediction, pred_score):
    cnf_matrix = confusion_matrix(y_true, y_prediction)
    try:
        tn, fp, fn, tp = cnf_matrix.ravel()
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f1_value = f1_score(y_true, y_prediction)
    accuracy = accuracy_score(y_true, y_prediction)

    # auc_value
    fpr, tpr, threshold = roc_curve(y_true, pred_score)
    auc_value = auc(fpr, tpr)

    return accuracy, f1_value, tp, fp, tn, fn, auc_value

# right result to report file
def write_report(result_scores):
    record_pd = pd.DataFrame(np.array(result_scores).reshape(1, -1), columns=RECORD_COLUMNS)
    return record_pd

