'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/6 
    Python Version: 3.6
'''
"""
Code to split the data into train and test dataset
"""
import configparser
import logging
import sys

import click
import ast
import pandas as pd
import os
from configparser import ExtendedInterpolation

sys.path.append('.')
from src.data.utils import read_raw_data, select_years, get_city_output_path, outer_concatenate, read_query_from_file


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('merged_file_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path())
@click.argument('valid_data_path', type=click.Path())
@click.argument('test_data_path', type=click.Path())
def extract_file(config_path, merged_file_path, train_data_path, valid_data_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('data train-test splitting')

    # save path exist make sure
    save_pardir = os.path.dirname(train_data_path)
    if not os.path.exists(save_pardir):
        os.makedirs(save_pardir)

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    city_list = ast.literal_eval(pars['global']['city'])
    train_years = ast.literal_eval(pars['global']['train_year'])
    valid_years = ast.literal_eval(pars['global']['valid_year'])
    test_years = ast.literal_eval(pars['global']['test_year'])
    # seed word list
    seed_path = pars['extract_search_trend']['term_list_path']
    seed_word_list = read_query_from_file(seed_path)

    seed_word_list.append('DATE')
    # pol label list
    label_columns = ast.literal_eval(pars['extract_pol_label']['y_column_name'])
    seed_word_list = label_columns + seed_word_list

    # concatenate the train and valid data
    x_train_all, x_valid_all, x_test_all = pd.DataFrame(columns=seed_word_list),\
                                           pd.DataFrame(columns=seed_word_list), pd.DataFrame(columns=seed_word_list)
    place_holder_df = pd.DataFrame(columns=seed_word_list)

    for city in city_list:
        input_single_file_path = get_city_output_path(merged_file_path, city)
        output_city_test_path = get_city_output_path(test_data_path, city)
        output_city_train_path = get_city_output_path(train_data_path, city)
        output_city_valid_path = get_city_output_path(valid_data_path, city)

        merged_data = read_raw_data(input_single_file_path)
        merged_data.index = pd.to_datetime(merged_data.DATE)

        train_data = select_years(merged_data, train_years)
        valid_data = select_years(merged_data, valid_years)
        test_data = select_years(merged_data, test_years)

        train_data = outer_concatenate(place_holder_df, train_data)
        valid_data = outer_concatenate(place_holder_df, valid_data)
        test_data = outer_concatenate(place_holder_df, test_data)

        # save single city data
        train_data.to_csv(output_city_train_path, index=False)
        valid_data.to_csv(output_city_valid_path, index=False)
        test_data.to_csv(output_city_test_path, index=False)

        # concatenate data
        x_train_all = outer_concatenate(x_train_all, train_data)
        x_valid_all = outer_concatenate(x_valid_all, valid_data)
        x_test_all = outer_concatenate(x_test_all, test_data)

    # drop all NAs columns
    # this is not correct since some of them might two cities => cause NAs
#     x_train_all.dropna(axis=1, how='all', inplace=True)
#     x_valid_all.dropna(axis=1, how='all', inplace=True)
#     x_test_all.dropna(axis=1, how='all', inplace=True)

    # create train.csv, test.csv to check existence
    x_train_all.to_csv(train_data_path, index=False)
    x_valid_all.to_csv(valid_data_path, index=False)
    x_test_all.to_csv(test_data_path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    extract_file()
