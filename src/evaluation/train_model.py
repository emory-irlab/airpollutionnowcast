import ast
import configparser
import logging
import os
import sys
from configparser import ExtendedInterpolation

import click

sys.path.append('.')
from src.evaluation.utils import process_features, get_model_from_config, get_feature_from_config, \
    get_feature_pars, if_create_filtered_dict


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('valid_data_path', type=click.Path(exists=True))
def extract_file(config_path, train_data_path, valid_data_path):
    logger = logging.getLogger(__name__)
    logger.info('train model from training data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    # global parameters
    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    model_type = pars['train_model']['model_type']
    features_array = ast.literal_eval(pars['train_model']['FEATURE'])
    # get feature_pars dict
    for index in range(0, len(features_array)):
        feature_pars = get_feature_pars(pars, index)
        # save input_data_path for dllstm model
        feature_pars['input_data_path'] = valid_data_path

        y_train, train_pol, train_phys, train_trend = process_features(train_data_path, seq_length, search_lag)
        y_valid, valid_pol, valid_phys, valid_trend = process_features(valid_data_path, seq_length, search_lag)

        ## check if dllstm model, create filtered dict
        train_trend, valid_trend = if_create_filtered_dict(feature_pars, train_trend, valid_trend)

        x_train, embedding_dim = get_feature_from_config(feature_pars, train_pol, train_phys, train_trend)
        x_valid, _ = get_feature_from_config(feature_pars, valid_pol, valid_phys, valid_trend)

        model = get_model_from_config(feature_pars, model_type, embedding_dim)

        # build model
        model.build_model()
        model.fit(x_train, x_valid, y_train, y_valid)

        save_model_path = feature_pars['save_model_path']
        if not os.path.exists(os.path.dirname(save_model_path)):
            os.makedirs(os.path.dirname(save_model_path))
        model.save(save_model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()


