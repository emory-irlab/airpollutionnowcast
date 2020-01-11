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

sys.path.append('.')
from src.data.utils import read_raw_data, select_years


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('merged_file_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path())
@click.argument('valid_data_path', type=click.Path())
@click.argument('test_data_path', type=click.Path())
def extract_file(config_path, merged_file_path, train_data_path, valid_data_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('data train-test splitting')

    pars = configparser.ConfigParser()
    pars.read(config_path)

    city_list = ast.literal_eval(pars['global']['city'])
    train_years = ast.literal_eval(pars['global']['train_year'])
    valid_years = ast.literal_eval(pars['global']['valid_year'])
    test_years = ast.literal_eval(pars['global']['test_year'])
    # seed word list
    seed_path = pars['extract_search_trend']['term_list_path']
    seed_word_list = list(set([k.lower() for k in pd.read_csv(seed_path, header=None)[0].values]))
    seed_word_list.append('DATE')
    # pol label list
    label_columns = ast.literal_eval(pars['extract_pol_label']['y_column_name'])
    seed_word_list = label_columns + seed_word_list

    input_single_file_name = '_' + os.path.basename(merged_file_path)

    # concatenate the train and valid data
    x_train_all, x_valid_all, x_test_all = pd.DataFrame(columns=seed_word_list),\
                                           pd.DataFrame(columns=seed_word_list), pd.DataFrame(columns=seed_word_list)

    for city in city_list:
        input_single_file_path = os.path.join(os.path.dirname(merged_file_path), city + input_single_file_name)

        merged_data = read_raw_data(input_single_file_path)
        merged_data.index = pd.to_datetime(merged_data.DATE)

        train_data = select_years(merged_data, train_years)
        valid_data = select_years(merged_data, valid_years)
        test_data = select_years(merged_data, test_years)

        # concatenate data
        x_train_all = pd.concat([x_train_all, train_data], ignore_index=True, sort=False)
        x_valid_all = pd.concat([x_valid_all, valid_data], ignore_index=True, sort=False)
        x_test_all = pd.concat([x_test_all, test_data], ignore_index=True, sort=False)

    # drop all NAs columns
    x_train_all.dropna(axis=1, how='all', inplace=True)
    x_valid_all.dropna(axis=1, how='all', inplace=True)
    x_test_all.dropna(axis=1, how='all', inplace=True)

    # create train.csv, test.csv to check existence
    x_train_all.to_csv(train_data_path, index=False)
    x_valid_all.to_csv(valid_data_path, index=False)
    x_test_all.to_csv(test_data_path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    extract_file()
