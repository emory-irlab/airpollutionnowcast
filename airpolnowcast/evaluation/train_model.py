import ast
import configparser
import logging
import os
import sys
sys.path.append('.')
from configparser import ExtendedInterpolation
import pandas as pd

import click
from airpolnowcast.evaluation.utils import get_model_from_config, get_feature_pars
from airpolnowcast.data.utils import read_query_from_file
from airpolnowcast.features.build_features import FeatureEngineer
import tensorflow as tf


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('valid_data_path', type=click.Path(exists=True))
def extract_file(config_path, train_data_path, valid_data_path):
    logger = logging.getLogger(__name__)
    logger.info('train model from training data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)
    # automatically get commit id from environment
    commit_id = os.popen('git rev-parse HEAD').read().replace('\n', '')
    pars['DEFAULT']['commit_id'] = commit_id

    # global parameters
    # seed word list
    seed_path = pars['extract_search_trend']['term_list_path']
    seed_word_list = read_query_from_file(seed_path)
    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    # features_array = ast.literal_eval(pars['train_model']['FEATURE'])
    use_feature = ast.literal_eval(pars['train_model']['use_feature'])

    # create object for feature engineer
    feature_engineer = FeatureEngineer()

    # get feature_pars dict
    for index in use_feature:
        feature_pars = get_feature_pars(pars, index)
        # get model_type
        model_type = feature_pars['model_type']
        save_model_path = feature_pars['save_model_path']
        if os.path.exists(save_model_path):
            logger.info("Model File Exist! Change Model Path\n")
            continue
        # save input_data_path for dllstm model
        feature_pars['input_data_path'] = valid_data_path

        y_train, train_pol, train_phys, train_trend = feature_engineer.feature_from_file(train_data_path, seq_length, search_lag)
        y_valid, valid_pol, valid_phys, valid_trend = feature_engineer.feature_from_file(valid_data_path, seq_length, search_lag)

        # check if dllstm model, create filtered dict
        train_trend, valid_trend = feature_engineer.match_query_order(feature_pars, train_trend, valid_trend, seed_word_list)

        x_train, embedding_dim = feature_engineer.create_feature_sequence(feature_pars, train_pol, train_phys, train_trend)
        x_valid, _ = feature_engineer.create_feature_sequence(feature_pars, valid_pol, valid_phys, valid_trend)

        model = get_model_from_config(feature_pars, model_type, embedding_dim)

        # build model
        model.build_model()
        model.fit(x_train, x_valid, y_train, y_valid)

        model_pardir = os.path.dirname(save_model_path)

        if not os.path.exists(model_pardir):
            os.makedirs(model_pardir)
        model.save(save_model_path)

        # save config file
        save_config_path = os.path.join(model_pardir, 'config.ini')
        with open(save_config_path, 'w') as configfile:
            pars.write(configfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    extract_file()


