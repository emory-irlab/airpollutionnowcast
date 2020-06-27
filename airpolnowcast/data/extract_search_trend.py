# -*- coding: utf-8 -*-
# from dotenv import find_dotenv, load_dotenv
import configparser
import logging
import sys
from pathlib import Path
from configparser import ExtendedInterpolation
import click
import pandas as pd
import os

sys.path.append('.')
from airpolnowcast.data.utils import read_global_pars, extract_from_raw_data

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def extract_search_trend(config_path, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim search data set from raw data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    city_list, years, season, abs_data_path = read_global_pars(pars)
    seed_path = pars['extract_search_trend']['term_list_path']
    name_pattern = pars['extract_search_trend']['name_pattern']
    search_data_path = pars['extract_search_trend']['search_data_path']

    # extract single city search data
    extract_from_raw_data(city_list, seed_path, output_filepath, abs_data_path, search_data_path, name_pattern, season, years)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    extract_search_trend()
