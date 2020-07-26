"""
Code to build feature from input csv
"""
import pandas as pd
import numpy as np
# import sys
# sys.path.append('.')
import random


# code to process data to x and y variables
def process_data(input_data):
    y = input_data.iloc[:, 0]
    pol_val = input_data.iloc[:, 1]
    date_col_index = input_data.columns.get_loc("DATE")
    trend_fea = input_data.iloc[:, 2:date_col_index]
    phys_fea = input_data.iloc[:, date_col_index+1:]
    trend_fea.index = input_data.iloc[:,date_col_index]
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
    :param lag: int:
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
    df_index_names = input_df.index
    input_df = np.array(input_df)
    embedding_dim = input_df.shape[1]
    reveserse_embeddings = input_df[::-1]
    lag_features = np.roll(reveserse_embeddings, lag, axis=0)
    for i in range(lag):
        na_embedding = np.array([rnd() for k in range(embedding_dim)])
        lag_features[i] = na_embedding
    lag_features = lag_features[::-1]
    return pd.DataFrame(lag_features, columns=df_column_names, index = df_index_names)


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






