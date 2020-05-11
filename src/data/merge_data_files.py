'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/6 
    Python Version: 3.6
'''
"""
Code to merge features
"""

import configparser
import logging
import sys

import click
import os
import pandas as pd
import ast

sys.path.append('.')
from src.data.utils import read_raw_data, normalize_column


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('pol_path', type=click.Path(exists=True))
@click.argument('search_path', type=click.Path(exists=True))
@click.argument('process_phys_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def extract_file(config_path, pol_path, search_path, process_phys_path, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making physical measurements features from raw data')

    pars = configparser.ConfigParser()
    pars.read(config_path)

    city_list = ast.literal_eval(pars['global']['city'])

    # re-direct search path 
    search_path = pars['merge_data_files'][pars['merge_data_files']['search_data_source']]
    
    input_pol_name = '_' + os.path.basename(pol_path)
    input_search_name = '_' + os.path.basename(search_path)
    input_phys_name = '_' + os.path.basename(process_phys_path)
    output_single_file_name = '_' + os.path.basename(output_filepath)

    for city in city_list:
        single_pol_path = os.path.join(os.path.dirname(pol_path), city + input_pol_name)
        single_search_path = os.path.join(os.path.dirname(search_path), city + input_search_name)
        single_phys_path = os.path.join(os.path.dirname(process_phys_path), city + input_phys_name)
        output_single_file_path = os.path.join(os.path.dirname(output_filepath), city + output_single_file_name)

        pol_data = read_raw_data(single_pol_path, add_datetime=True)
        # norm the pollution values
        pol_data.iloc[:, 1] = normalize_column(pol_data.iloc[:, 1])

        search_data = read_raw_data(single_search_path, add_datetime=True, norm_col=True)
        process_phys_data = read_raw_data(single_phys_path, add_datetime=True, norm_col=True)

        # merge pollution dataframe with label df
        output_df = pd.merge(pol_data, search_data, left_index=True, right_index=True, how='left',
                                indicator=False)

        output_df['DATE'] = output_df.index.format()
        output_df = pd.merge(output_df, process_phys_data, left_index=True, right_index=True, how='left',
                             indicator=False)
        output_df.to_csv(output_single_file_path, index=False)

    # create merged.csv to check existence
    open(output_filepath, 'w').close()
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    extract_file()
