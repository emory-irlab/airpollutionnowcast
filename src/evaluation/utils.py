'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/7 
    Python Version: 3.6
'''

import os
import pickle
import sys

sys.path.append('.')
from src.data import read_raw_data
from src.features.build_features import process_data, get_pol_value_series, lag_search_features, get_feature_array
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score, \
    average_precision_score, precision_recall_curve
from src.models.rf import RandomForestModel
from src.models.lr import LRModel
from src.models.composed_lstm import ComposedLSTM
from src.models.lstm import LSTMModel
from src.models.dict_learner_sensor import ComposedDLLSTMModel
from src.models.dict_learner import DLLSTMModel
from src.models.wvlstm import WVLSTM
from src.models.multitask_learning import ComposedMTLSTM
from src.models.multitask_lstm import MTLSTM
from src.models.multitask_dllstm import MTDLLSTM
import ast
import logging

RECORD_COLUMNS = ['city', 'model', 'feature', 'is_two_branch', 'search_lag', 'city_fine_tuning', 'accuracy', 'F1 score', 'true positives',
                  'false positives', 'true negatives',
                  'false negatives', 'AUC', 'AP', 'Interpolated-AP']


# get random forest model
def get_rf_model(pars):
    # parameters for rf model
    n_estimators = pars['n_estimators']
    max_depth = pars['max_depth']
    model = RandomForestModel(n_estimators, max_depth)
    return model


def get_lstm_model(feature_pars, embedding_dim, model_type):
    kwargs = {'seq_length': feature_pars['seq_length'], 'embedding_dim': embedding_dim,
              'learning_rate': feature_pars['learning_rate'],
              'batch_size': feature_pars['batch_size'],
              'patience': feature_pars['patience'],
              'log_dir': feature_pars['log_dir']}
    log_dir = feature_pars['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    two_branch = feature_pars['is_two_branch']

    if model_type == 'dllstm':
        if two_branch:
            model = ComposedDLLSTMModel(**kwargs)
        else:
            model = DLLSTMModel(**kwargs)  # need to implement trend only DLLSTMModel
            pass
        # save necessary dict path for dllstm model
        model.get_glove_and_intent_path(feature_pars)
    elif model_type == 'lstm':
        if two_branch:
            model = ComposedLSTM(**kwargs)
        else:
            model = LSTMModel(**kwargs)
    elif model_type == 'mtlstm':
        if two_branch:
            model = ComposedMTLSTM(**kwargs)
        else:
            model = MTLSTM(**kwargs)
    elif model_type == 'wvlstm':
        if two_branch:
            model = None
        else:
            model = WVLSTM(**kwargs)
        model.get_glove_and_intent_path(feature_pars)
    elif model_type == 'mtdllstm':
        if two_branch:
            pass
        else:
            model = MTDLLSTM(**kwargs)
        # save necessary dict path for dllstm model
        model.get_glove_and_intent_path(feature_pars)
    return model


def get_model_from_config(feature_pars, model_type, embedding_dim):
    if model_type == 'rf':
        model = get_rf_model(feature_pars)
    elif model_type == 'lr':
        tuned_parameters = {'Cs': list(np.power(10.0, np.arange(-10, 5))),
                            'l1_ratios': list(np.power(10.0, np.arange(-10, 0)))}
        model = LRModel(tuned_parameters)
    elif model_type in ['lstm', 'dllstm', 'mtlstm', 'mtdllstm', 'wvlstm']:
        model = get_lstm_model(feature_pars, embedding_dim, model_type)
    return model


def get_feature_pars(pars, index):
    feature_pars = dict()
    # feature parameters
    feature_pars['is_two_branch'] = (ast.literal_eval(pars['train_model']['two_branch'])[index] == 'yes')
    feature_pars['first_branch'] = ast.literal_eval(pars['train_model']['first_branch'])[index]
    feature_pars['second_branch'] = ast.literal_eval(pars['train_model']['second_branch'])[index]
    features_array = ast.literal_eval(pars['train_model']['FEATURE'])
    feature_pars['feature'] = features_array[index]

    # model parameters
    feature_pars['seq_length'] = int(pars['train_model']['seq_length'])
    feature_pars['search_lag'] = int(pars['train_model']['search_lag'])
    feature_pars['model_type'] = ast.literal_eval(pars['train_model']['model_type'])[index]

    # path parameters
    feature_pars['log_dir'] = os.path.join(pars['train_model']['log_dir'],
                                           feature_pars['model_type'],
                                           feature_pars['feature'])
    feature_pars['save_model_path'] = os.path.join(pars['train_model']['save_model_path'],
                                                   feature_pars['model_type'],
                                                   feature_pars['feature'] + '.h5')
    # path for fine_tuning models
    feature_pars['save_fine_tuning_path'] = os.path.join(pars['predict_fine_tuning']['save_fine_tuning_path'],
                                                         feature_pars['model_type'],
                                                         feature_pars['feature'] + '.h5')

    # path for dlstm
    feature_pars['intent_dict_path'] = pars['DLLSTM']['intent_dict_path']
    feature_pars['filtered_dict_path'] = pars['DLLSTM']['filtered_dict_path']
    feature_pars['current_word_path'] = pars['DLLSTM']['current_word_path']
    feature_pars['search_terms_dict_path'] = pars['DLLSTM']['search_terms_dict_path']
    feature_pars['seed_word_path'] = pars['DLLSTM']['seed_word_path']

    # model parameters for lstm
    feature_pars['learning_rate'] = float(pars['train_model']['learning_rate'])
    feature_pars['batch_size'] = int(pars['train_model']['batch_size'])
    feature_pars['patience'] = int(pars['train_model']['patience'])

    # model parameters for rf
    feature_pars['n_estimators'] = ast.literal_eval(pars['train_model']['n_estimators'])
    feature_pars['max_depth'] = ast.literal_eval(pars['train_model']['max_depth'])

    # assert lens equal
    len_feature = len(features_array)
    len_two_branch = len(ast.literal_eval(pars['train_model']['two_branch']))
    len_first_branch = len(ast.literal_eval(pars['train_model']['first_branch']))
    len_second_branch = len(ast.literal_eval(pars['train_model']['second_branch']))

    lists = [len_feature, len_two_branch, len_first_branch, len_second_branch]
    it = iter(lists)
    the_len = next(it)
    if not all(l == the_len for l in it):
        raise ValueError('Not all input feature list have the same length!')

    return feature_pars


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
    ap_value = average_precision_score(y_true, pred_score)
    intpol_ap_value = naive_interpolated_precision(y_true, pred_score)

    return [accuracy, f1_value, tp, fp, tn, fn, auc_value, ap_value, intpol_ap_value]


# right result to report file
def write_report(result_scores, record_pd, index):
    record_pd.loc[index, RECORD_COLUMNS] = result_scores
    return record_pd


"""
Evaluation metrics
"""


def naive_interpolated_precision(y_true, y_scores):
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    interp_precisions = []

    # the final point
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    for recall_point in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        interp_precisions.append(precisions[recalls >= recall_point].max())

    return np.mean(interp_precisions)


