'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/7 
    Python Version: 3.6
'''

import sys
import os
import pickle
from typing import Set, Dict, List, Any, Union, Tuple

sys.path.append('.')
from src.data import read_raw_data
from src.features.build_features import process_data, get_pol_value_series, lag_search_features, get_feature_array
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from src.models.rf import RandomForestModel
from src.models.composed_lstm import ComposedLSTM
from src.models.lstm import LSTMModel
from src.models.dict_learner_sensor import DLLSTMModel
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


def get_two_branch_feature(seq_length, first_branch, second_branch):
    x_train_sensor, embedding_sensor = get_feature_array(first_branch, seq_length)
    x_train_trend, embedding_trend = get_feature_array(second_branch, seq_length)

    x_train = (x_train_sensor, x_train_trend)
    embedding = (embedding_sensor, embedding_trend)
    return x_train, embedding


def get_lstm_model(pars, embedding_dim, model_type):
    seq_length = int(pars['train_model']['seq_length'])
    learning_rate = float(pars['train_model']['learning_rate'])
    batch_size = int(pars['train_model']['batch_size'])
    patience = int(pars['train_model']['patience'])
    log_dir = pars['train_model']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    two_branch = pars['train_model'].getboolean('two_branch')
    if not two_branch:
        model = LSTMModel(seq_length, embedding_dim, learning_rate, batch_size, patience, log_dir)
    else:
        if model_type == 'lstm':
            model = ComposedLSTM(seq_length, embedding_dim, learning_rate, batch_size, patience, log_dir)
        elif model_type == 'dllstm':
            model = DLLSTMModel(seq_length, embedding_dim, learning_rate, batch_size, patience, log_dir)
            # save necessary dict path for dllstm model
            model.get_glove_and_intent_path(pars)
    return model


def get_model_from_config(pars, model_type, embedding_dim):
    if model_type == 'rf':
        model = get_rf_model(pars)
    elif model_type in ['lstm', 'dllstm']:
        model = get_lstm_model(pars, embedding_dim, model_type)
    return model


def get_feature_from_config(pars, model_type, train_pol, train_phys, train_trend, seq_length):
    if model_type in ['lstm', 'dllstm']:
        two_branch = pars['train_model'].getboolean('two_branch')
        if two_branch:
            first_branch = ast.literal_eval(pars['train_model']['first_branch'])
            second_branch = ast.literal_eval(pars['train_model']['second_branch'])
            first_feature = get_feature_from_code(first_branch, train_pol, train_phys, train_trend)
            second_feature = get_feature_from_code(second_branch, train_pol, train_phys, train_trend)
            x_train, embedding_dim = get_two_branch_feature(seq_length, first_feature, second_feature)
    else:
        code_feature: List[int] = ast.literal_eval(pars['train_model']['code_feature'])
        feature_list = get_feature_from_code(code_feature, train_pol, train_phys, train_trend)
        x_train, embedding_dim = get_feature_array(feature_list, seq_length)
    return x_train, embedding_dim


def get_feature_from_code(feature_code, train_pol, train_phys, train_trend):
    """

    :param feature_code: list(int0
        shape: 3 (pol, phys, trend0
        0 for not including, 1 for including
    :param train_pol:
    :param train_phys:
    :param train_trend:
    :return: list(np.array)
    """
    output_feature = []
    list_feature = [train_pol, train_phys, train_trend]

    for i in range(3):
        if feature_code[i] == 1:
            output_feature.append(list_feature[i])

    return output_feature



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


"""
Function for DLLSTM
"""


def generate_dllstm_filtered_dict(pars):
    # get common terms
    current_word_path = pars['DLLSTM']['current_word_path']
    # check for filtered_dict_path
    filtered_dict_path = pars['DLLSTM']['filtered_dict_path']
    search_terms_dict_path = pars['DLLSTM']['search_terms_dict_path']
    seed_word_path = pars['DLLSTM']['seed_word_path']
    input_data_path = pars['DLLSTM']['input_data_path']
    search_volume_df = read_raw_data(input_data_path)
    with open(search_terms_dict_path, 'rb') as f:
        a = pickle.load(f)
    SEED = True
    if SEED:
        seed_word_path = seed_word_path
        seed_word_list = [k.lower() for k in pd.read_csv(seed_word_path, header=None)[0].values]
        terms = []
        for c in seed_word_list:
            if c in a.keys() and c in search_volume_df.columns:
                terms.append(c)
    else:
        terms = []
        for c in a.keys():
            if c in search_volume_df.columns:
                terms.append(c)
    # print(search_volume_df.shape, len(terms))
    # Store a dictionary of terms currently in use and their glove embeddings.
    terms = list(set(terms))
    with open(filtered_dict_path, 'wb') as f:
        pickle.dump({k: a[k] for k in terms}, f)
    with open(current_word_path, 'wb') as f:
        pickle.dump(terms, f)



