'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/5 
    Python Version: 3.6
'''
import os
import pandas as pd
import datetime as dt
import ast
import numpy as np


def read_global_pars(pars):
    city_list = ast.literal_eval(pars['global']['city'])
    train_years = ast.literal_eval(pars['global']['train_year'])
    valid_years = ast.literal_eval(pars['global']['valid_year'])
    test_years = ast.literal_eval(pars['global']['test_year'])
    years = train_years + valid_years + test_years
    season = pars['global']['season']
    abs_data_path = pars['global']['abs_data_path']
    return city_list, years, season, abs_data_path


def read_raw_data(raw_file_path, add_datetime=False, norm_col=False):
    trend_data = pd.read_csv(raw_file_path)
    if add_datetime:
        trend_data = add_datetime_index(trend_data)
    if norm_col:
        trend_data = normalize_column(trend_data)
    return trend_data


# generate complete file path given data_path, city and name_pattern
def generate_file_path(data_path, name_pattern, city):
    """

    :param data_path: str
        path to the data folder
    :param name_pattern: str
    :param city: str
    :return: str
    """

    raw_file_name = name_pattern.replace('CITY', city)
    raw_file_path = os.path.join(data_path, raw_file_name)
    return raw_file_path


# add datetime index for input dataframe
def add_datetime_index(input_df):
    """

    :param input_df: pd.DataFrame
    :return: pd.DataFrame
    """

    input_df.rename(columns={input_df.columns[0]: "date"}, inplace=True)
    input_df.index = pd.to_datetime(input_df['date'])
    input_df.drop(['date'], axis=1, inplace=True)
    return input_df


# select single year data
def select_single_year(input_df, single_year):
    """

    :param input_df: pd.DataFrame
    :param single_year: int
    :return: list(int)
    """
    start = input_df.index.searchsorted(dt.date(single_year, 1, 1))
    end = input_df.index.searchsorted(dt.date(single_year, 12, 31))
    if dt.date(single_year, 12, 31) not in input_df.index:
        end -= 1
    single_year_index = [k for k in range(start, end + 1)]
    return single_year_index


# select data according to selected years
def select_years(input_df, years):
    """

    :param input_df: pd.DataFrame
    :param years: list(int)
    :return: pd.DataFrame
    """
    selected_days = []
    for single_year in years:
        single_year_index = select_single_year(input_df, single_year)
        selected_days = selected_days + single_year_index

    output_df = input_df.iloc[selected_days, :]
    return output_df


# the cold or warm seasons
def isColdWarm(input_df):
    dates = input_df['date']
    # get month, day
    fields = dates.split("-")
    if len(fields) < 3:
        return input_df

    month = int(fields[1])
    if (month >= 3) and (month <= 10):
        input_df['isWarm'] = True
        return input_df
    else:
        input_df['isCold'] = True
        return input_df


# select data from cold or warm seasons
def select_season(input_df, season):
    if season == 'all':
        return input_df
    else:
        out_df = input_df.copy()
        out_df['date'] = out_df.index.format()
        out_df['isCold'] = False
        out_df['isWarm'] = False
        out_df = out_df.apply(isColdWarm, axis=1)

        if season == 'warm':
            out_df = out_df[out_df.isWarm == True]
        elif season == 'cold':
            out_df = out_df[out_df.isCold == True]

        out_df.drop(['date', 'isCold', 'isWarm'], axis=1, inplace=True)
        return out_df


# def select data from word list
def get_common_terms(input_df, word_list):
    input_terms = list(input_df.columns)
    common_terms = list(set(input_terms).intersection(set(word_list)))
    return common_terms


# write a common function for select years and season
def common_selection(abs_data_path, input_file_path, name_pattern, city, season, years):
    input_file_path = os.path.join(abs_data_path, input_file_path)
    raw_file_path = generate_file_path(input_file_path, name_pattern, city)
    output_data = read_raw_data(raw_file_path)
    output_data = add_datetime_index(output_data)
    output_data = select_years(output_data, years)
    output_data = select_season(output_data, season)
    return output_data


# a shared function for extract information from raw data
def extract_from_raw_data(city_list, raw_select_columns, output_filepath, abs_data_path, search_data_path, name_pattern,
                          season, years):
    """
    :param city_list:
    :param raw_select_columns:
    :param output_filepath:
    :param abs_data_path:
    :param search_data_path:
    :param name_pattern:
    :param season:
    :param years:
    :return:
    """

    single_file_name = '_' + os.path.basename(output_filepath)

    # mark whether it's trend data
    is_trend = False

    # if select columns not list
    if not isinstance(raw_select_columns, list):
        # Get seed_word_list
        seed_word_list = [k.lower() for k in pd.read_csv(raw_select_columns, header=None)[0].values]
        # mark is_trend
        is_trend = True

    for city in city_list:
        single_file_path = os.path.join(os.path.dirname(output_filepath), city + single_file_name)
        trend_data = common_selection(abs_data_path, search_data_path, name_pattern, city, season, years)
        if not isinstance(raw_select_columns, list):
            select_columns = get_common_terms(trend_data, seed_word_list)
        else:
            select_columns = raw_select_columns[:]

        extract_single_city_data(select_columns, single_file_path, trend_data, is_trend)
    # create search.csv to check existence
    open(output_filepath, 'w').close()


"""
Utils for extract_search_trend.py
"""


def extract_single_city_data(select_columns, single_file_path, trend_data, is_trend):
    trend_data = trend_data[select_columns]

    if is_trend:
        # replace those smaller than 0.1 as nan
        log_df = trend_data.copy()
        log_df = log_df.apply(lambda x: np.where(x < 0.1, [np.nan for k in range(len(x))], x))

        # remove most NA columns
        drop_constant_name = list(log_df.loc[:, log_df.count() < 18].columns)
        # drop same values all time terms
        drop_constant_name.extend(list(log_df.loc[:, log_df.std() < 2].columns))
        if len(drop_constant_name) != 0:
            print('========Drop Most NAs========')
            print(drop_constant_name)

        trend_data = trend_data.drop(set(drop_constant_name), axis=1)
    trend_data.to_csv(single_file_path)


"""
Utils for extract_pol_label.py
"""


"""
Utils for process_phys_feature.py
"""


def add_quadratic_terms(phys_data, TEMP_MAX_COLUMN_NAME, TEMP_MEAN_COLUMN_NAME, RH_MEAN_COLUMN_NAME):
    phys_data['quad_temp'] = phys_data[TEMP_MAX_COLUMN_NAME] ** 2
    phys_data['cubic_temp'] = phys_data[TEMP_MAX_COLUMN_NAME] ** 3
    temp_celsius = (phys_data[TEMP_MEAN_COLUMN_NAME] - 32) * 0.5556
    rh_mean = phys_data[RH_MEAN_COLUMN_NAME]

    phys_data['dew_point'] = 243.04 * (
            np.log(rh_mean / 100) + ((17.625 * temp_celsius) / (243.04 + temp_celsius))) / (
                                     17.625 - np.log(rh_mean / 100
                                                     ) - ((17.625 * temp_celsius) / (243.04 + temp_celsius)))
    phys_data['quad_dew'] = phys_data['dew_point'] ** 2
    phys_data['cubic_dew'] = phys_data['dew_point'] ** 3
    return phys_data


## coefficients for heat index
c1, c2, c3 = -42.379, 2.04901523, 10.14333127
c4, c5, c6 = -0.22475541, -6.83783e-3, -5.481717e-2
c7, c8, c9 = 1.22874e-3, 8.5282e-4, -1.99e-6


def add_heat_index(process_phys, TEMP_MEAN_COLUMN_NAME, RH_MEAN_COLUMN_NAME):
    T = process_phys[TEMP_MEAN_COLUMN_NAME]
    R = process_phys[RH_MEAN_COLUMN_NAME]
    process_phys['heat'] = c1 + c2 * T + c3 * R + c4 * T * R + c5 * T * T + c6 * R * R + c7 * T * T * R + c8 * T * R * R \
                           + c9 * T * T * R * R
    return process_phys


"""
Utils for merge_data_files.py
"""


# normalize the column
def normalize_column(input_df):
    output_df = (input_df - input_df.mean()) / input_df.std()
    return output_df

"""
Utils for train_test_split.py
"""


# get city output path
def get_city_output_path(template_file_path, city):
    file_name = '_' + os.path.basename(template_file_path)
    output_path = os.path.join(os.path.dirname(template_file_path), city + file_name)
    return output_path


# get inner join common columns
def inner_concatenate(x_train_all, train_data):
    x_train_all = pd.concat([x_train_all, train_data], join='inner', ignore_index=True, sort=False)
    return x_train_all