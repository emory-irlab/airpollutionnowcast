"""
Code to build feature from input csv
"""
import pandas as pd
import numpy as np
# import sys
# sys.path.append('.')
import random
from typing import List
import os
import pickle
from src.data.utils import read_raw_data


class FeatureEngineer(object):

    @staticmethod
    def feature_from_file(train_data_path, seq_length, search_lag):
        y_train, train_pol, train_phys, train_trend = process_features(train_data_path, seq_length, search_lag)
        return y_train, train_pol, train_phys, train_trend

    @staticmethod
    def match_query_order(feature_pars, train_trend, valid_trend, seed_word_list):
        # check if dllstm model, create filtered dict
        train_trend, valid_trend = if_create_filtered_dict(feature_pars, train_trend, valid_trend, seed_word_list)
        return train_trend, valid_trend

    @staticmethod
    def create_feature_sequence(feature_pars, train_pol, train_phys, train_trend):
        x_train, embedding_dim = get_feature_from_config(feature_pars, train_pol, train_phys, train_trend)
        return x_train, embedding_dim


# code to process data to x and y variables
def process_data(input_data):
    y = input_data.iloc[:, 0]
    pol_val = input_data.iloc[:, 1]
    date_col_index = input_data.columns.get_loc("DATE")
    trend_fea = input_data.iloc[:, 2:date_col_index]
    phys_fea = input_data.iloc[:, date_col_index+1:]
    return y, pol_val, trend_fea, phys_fea


# change input pollution value to time series
def get_pol_value_series(data, seq_length):
    """
    :param data: pd.DataFrame
        shape N*1
        N: number of days
        1: pollutant concentration
    :param seq_length: int
    :return: np.array
        shape N*seq_length
        seq_length: previous seq_length days
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, seq_length + 1)]
    df = pd.concat(columns, axis=1)
    # fill NAs with 0
    df.fillna(0, inplace=True)
    supervised_values = np.array(df)

    return supervised_values


# apply random values to fill nas
def rnd():
    exp = random.randint(-10, -5)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp


# apply lags to search features
def lag_search_features(input_df, lag):
    """

    :param input_df: pd.DataFrame
        the search feature input data
        shape: N*M
        N: number of days
        M: number of search terms
    :param lag: int
        lag of days applied ont search data
    :return: pd.DataFrame
        shape: N*M
        for day i, we have the info of day i+lag (later)
    """
    # remove NAs for input_df with small values
    input_df = input_df.apply(lambda x:
                                np.where(x.isnull(), [rnd() for k in range(len(x))], x))
    # record column names
    df_column_names = input_df.columns
    input_df = np.array(input_df)
    embedding_dim = input_df.shape[1]
    reveserse_embeddings = input_df[::-1]
    lag_features = np.roll(reveserse_embeddings, lag, axis=0)
    for i in range(lag):
        na_embedding = np.array([rnd() for k in range(embedding_dim)])
        lag_features[i] = na_embedding
    lag_features = lag_features[::-1]
    return pd.DataFrame(lag_features, columns=df_column_names)


# generate sequence input features for LSTM training
def generate_input_sequence(input_array, seq_length):
    """

    :param input_array: np.array
        shape: N*P
        N: number of days
        P: number of features for day i
    :param seq_length: int
        sequence length for LSTM model
    :return: np.array
        shape: N*seq_length*P
    """
    embedding_dim = input_array.shape[1]
    input_embedding = []
    for i in range(len(input_array)):
        input_series = []
        for days_index in range(i - seq_length + 1, i + 1):
            if days_index >= 0:
                day_embedding = input_array[days_index]
            else:
                na_vec = np.array([rnd() for i in range(embedding_dim)])
                day_embedding = na_vec

            input_series.append(day_embedding)
        input_embedding.append(np.array(input_series))
    input_embedding = np.array(input_embedding)
    return input_embedding


def get_feature_array(final_feature_list, seq_length):
    train_feas_per_day = np.concatenate(final_feature_list, axis=1)
    feature_array = generate_input_sequence(train_feas_per_day, seq_length)

    embedding_dim = feature_array.shape[2]
    return feature_array, embedding_dim


# Functions to use in train_model
# read data and process to features
def process_features(train_data_path, seq_length, search_lag):

    train_data = read_raw_data(train_data_path)
    y_data, pol_val, trend_fea, phys_fea = process_data(train_data)
    processed_pol = get_pol_value_series(pol_val, seq_length)
    processed_trend = lag_search_features(trend_fea, search_lag)
    phys_fea = lag_search_features(phys_fea, 0)
    # fill NAs with 0 for phys_fea
    phys_fea.fillna(0, inplace=True)
    process_phys = np.array(phys_fea)
    return y_data, processed_pol, process_phys, processed_trend


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


def get_two_branch_feature(seq_length, first_branch, second_branch):
    x_train_sensor, embedding_sensor = get_feature_array(first_branch, seq_length)
    x_train_trend, embedding_trend = get_feature_array(second_branch, seq_length)

    x_train = (x_train_sensor, x_train_trend)
    embedding = (embedding_sensor, embedding_trend)
    return x_train, embedding


def get_feature_from_config(pars, train_pol, train_phys, train_trend):
    is_two_branch = pars['is_two_branch']
    seq_length = pars['seq_length']

    if is_two_branch:
        first_feature = get_feature_from_code(pars['first_branch'], train_pol, train_phys, train_trend)
        second_feature = get_feature_from_code(pars['second_branch'], train_pol, train_phys, train_trend)
        x_train, embedding_dim = get_two_branch_feature(seq_length, first_feature, second_feature)
    else:
        code_feature: List[int] = pars['first_branch']
        feature_list = get_feature_from_code(code_feature, train_pol, train_phys, train_trend)
        x_train, embedding_dim = get_feature_array(feature_list, seq_length)
    return x_train, embedding_dim


"""
Function for DLLSTM
"""


def generate_dllstm_filtered_dict(pars):
    # get common terms
    current_word_path = pars['current_word_path']
    # check for filtered_dict_path
    filtered_dict_path = pars['filtered_dict_path']
    search_terms_dict_path = pars['search_terms_dict_path']
    seed_word_path = pars['seed_word_path']
    input_data_path = pars['input_data_path']
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


def if_create_filtered_dict(feature_pars, train_trend, valid_trend, seed_word_list):
    model_type = feature_pars['model_type']

    if model_type in ['dllstm', 'mtdllstm']:
        # check for filtered_dict_path
        filtered_dict_path = feature_pars['filtered_dict_path']
        # get common terms
        current_word_path = feature_pars['current_word_path']
        if not (os.path.exists(filtered_dict_path) and os.path.exists(current_word_path)):
            generate_dllstm_filtered_dict(feature_pars)
        with open(current_word_path, 'rb') as f:
                common_terms = pickle.load(f)
#         print("common_terms: ")
#         print(common_terms[:5])
#         print()
#         print(train_trend.columns)
        train_trend = train_trend[common_terms]
        valid_trend = valid_trend[common_terms]
    else:
        train_trend = train_trend[seed_word_list]
        valid_trend = valid_trend[seed_word_list]

    return train_trend, valid_trend


if __name__ == "__main__":
    print("good")









