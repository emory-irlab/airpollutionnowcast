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
from src.data.utils import get_shuffle_index, year_column, get_shuffle_split, train_label


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
    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    # TODO: fix the error of pol_back_days
    pol_back_days = int(pars['train_model']['pol_back_days'])
    # features_array = ast.literal_eval(pars['train_model']['FEATURE'])
    use_feature = ast.literal_eval(pars['train_model']['use_feature'])
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
        feature_pars['input_data_path'] = train_data_path

        y_train, train_pol, train_phys, train_trend = process_features(train_data_path, search_lag, pol_back_days,
                                                                       train_label)
        # y_valid, valid_pol, valid_phys, valid_trend = process_features(valid_data_path, search_lag, pol_back_days, train_label)

        # check if dllstm model, create filtered dict
        # train_trend, valid_trend = if_create_filtered_dict(feature_pars, train_trend, valid_trend)

        x_train, embedding_dim = get_feature_from_config(feature_pars, train_pol, train_phys, train_trend)
        # x_valid, _ = get_feature_from_config(feature_pars, valid_pol, valid_phys, valid_trend)

        model = get_model_from_config(feature_pars, model_type, embedding_dim)

        # shuffle cross-validation
        tr_index, va_index = get_shuffle_index(y_train[year_column])
        if model_type not in ['rf', 'lr']:
            y_train.drop([year_column], axis=1, inplace=True)

        x_tr, x_va = get_shuffle_split(x_train, tr_index, va_index)
        y_tr, y_va = get_shuffle_split(y_train, tr_index, va_index)

        # build model
        model.build_model()
        model.fit(x_tr, x_va, y_tr, y_va)

        model_pardir = os.path.dirname(save_model_path)

        if not os.path.exists(model_pardir):
            os.makedirs(model_pardir)
        model.save(save_model_path)

    # save config file
    save_config_path = os.path.join(pars['train_model']['save_model_path'], 'config.ini')
    with open(save_config_path, 'w') as configfile:
        pars.write(configfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()
