import configparser
import logging
import os
import sys
from configparser import ExtendedInterpolation

import click
import pickle

sys.path.append('.')
from src.evaluation.utils import process_features, get_rf_model, get_two_branch_feature, get_lstm_model,\
    get_model_from_config, get_feature_from_config, generate_dllstm_filtered_dict
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
    # save input_data_path for dllstm model
    pars['DLLSTM']['input_data_path'] = valid_data_path

    y_train, train_pol, train_phys, train_trend = process_features(train_data_path, seq_length, search_lag)
    y_valid, valid_pol, valid_phys, valid_trend = process_features(valid_data_path, seq_length, search_lag)

    # design for dllstm model
    if model_type == 'dllstm':
        # check for filtered_dict_path
        filtered_dict_path = pars['DLLSTM']['filtered_dict_path']
        # get common terms
        current_word_path = pars['DLLSTM']['current_word_path']
        if not os.path.exists(filtered_dict_path):
            generate_dllstm_filtered_dict(pars)
        with open(current_word_path, 'rb') as f:
            common_terms = pickle.load(f)
        train_trend = train_trend[common_terms]
        valid_trend = valid_trend[common_terms]

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


