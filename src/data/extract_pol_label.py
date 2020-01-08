'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/6 
    Python Version: 3.6
'''

import configparser
import logging
import sys

import click
import ast
import os

sys.path.append('.')
from src.data.utils import read_global_pars, extract_from_raw_data


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def extract_file(config_path, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making interim pol data set from raw data')

    pars = configparser.ConfigParser()
    pars.read(config_path)

    city_list, years, season, abs_data_path = read_global_pars(pars)
    name_pattern = pars['extract_pol_label']['name_pattern']
    pol_data_path = pars['extract_pol_label']['pol_data_path']
    # select pol_type and label criteria
    y_column = ast.literal_eval(pars['extract_pol_label']['y_column_name'])

    # extract single city search data
    extract_from_raw_data(city_list, y_column, output_filepath, abs_data_path, pol_data_path, name_pattern, season, years)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()
