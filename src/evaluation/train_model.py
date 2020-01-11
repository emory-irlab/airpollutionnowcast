import configparser
import logging
import os
import sys
from configparser import ExtendedInterpolation

import click

sys.path.append('.')
from src.evaluation.utils import process_features, get_rf_model, get_two_branch_feature, get_lstm_model,\
    get_model_from_config, get_feature_from_config
from src.features.build_features import get_feature_array



@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('valid_data_path', type=click.Path(exists=True))
def extract_file(config_path, train_data_path, valid_data_path):
    logger = logging.getLogger(__name__)
    logger.info('train model from training data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    model_type = pars['train_model']['model_type']
    save_model_path = pars['train_model']['save_model_path']

    y_train, train_pol, train_phys, train_trend = process_features(train_data_path, seq_length, search_lag)
    y_valid, valid_pol, valid_phys, valid_trend = process_features(valid_data_path, seq_length, search_lag)

    x_train, embedding_dim = get_feature_from_config(pars, model_type, train_pol, train_phys, train_trend, seq_length)
    x_valid, _ = get_feature_from_config(pars, model_type, valid_pol, valid_phys, valid_trend, seq_length)

    model = get_model_from_config(pars, model_type, embedding_dim)

    # build model
    model.build_model()
    model.fit(x_train, x_valid, y_train, y_valid)

    if not os.path.exists(os.path.dirname(save_model_path)):
        os.makedirs(os.path.dirname(save_model_path))
    model.save(save_model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()


