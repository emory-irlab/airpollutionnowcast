'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/6 
    Python Version: 3.6
'''
"""
Code to transform physical measurement data
"""

import configparser
import logging
import sys

import click
import ast
import os

sys.path.append('.')
from src.data.utils import read_raw_data, add_quadratic_terms, add_heat_index, create_folder_exist
from configparser import ExtendedInterpolation

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('phys_file_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def extract_file(config_path, phys_file_path, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making interim physical measurements data from raw data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    # create folder if not exit
    create_folder_exist(os.path.dirname(output_filepath))
    city_list = ast.literal_eval(pars['global']['city'])

    # column names for physical measurement features
    TEMP_MAX_COLUMN_NAME = pars['DEFAULT']['TEMP_MAX_COLUMN_NAME']
    TEMP_MEAN_COLUMN_NAME = pars['DEFAULT']['TEMP_MEAN_COLUMN_NAME']
    RH_MEAN_COLUMN_NAME = pars['DEFAULT']['TEMP_MEAN_COLUMN_NAME']

    input_single_file_name = '_' + os.path.basename(phys_file_path)
    output_single_file_name = '_' + os.path.basename(output_filepath)

    for city in city_list:
        input_single_file_path = os.path.join(os.path.dirname(phys_file_path), city + input_single_file_name)
        output_single_file_path = os.path.join(os.path.dirname(output_filepath), city + output_single_file_name)
        phys_data = read_raw_data(input_single_file_path)

        process_phys = add_quadratic_terms(phys_data, TEMP_MAX_COLUMN_NAME, TEMP_MEAN_COLUMN_NAME, RH_MEAN_COLUMN_NAME)
        # process_phys = add_heat_index(process_phys, TEMP_MEAN_COLUMN_NAME, RH_MEAN_COLUMN_NAME)

        # index is int instead of datetime, this index set to false
        process_phys.to_csv(output_single_file_path, index=False)

    # create process_phys.csv to check existence
    open(output_filepath, 'w').close()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()
