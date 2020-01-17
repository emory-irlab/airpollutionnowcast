import ast
import configparser
import logging
import os
import pickle
import sys
from configparser import ExtendedInterpolation

import click

sys.path.append('.')
from src.evaluation.utils import process_features, result_stat, write_report, get_feature_from_config,\
    get_model_from_config, get_feature_pars


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('test_data_path', type=click.Path(exists=True))
def extract_file(config_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('predict the testing data')

    pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    pars.read(config_path)

    # global parameters
    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    model_type = pars['train_model']['model_type']
    features_array = ast.literal_eval(pars['train_model']['FEATURE'])
    # report path
    report_path = pars['predict_model']['report_path']

    # get feature_pars dict
    for index in range(0, len(features_array)):
        feature_pars = get_feature_pars(pars, index)
        # save input_data_path for dllstm model
        feature_pars['input_data_path'] = test_data_path
        y_test, test_pol, test_phys, test_trend = process_features(test_data_path, seq_length, search_lag)

        # design for dllstm model
        if model_type == 'dllstm':
            # get common terms
            current_word_path = feature_pars['current_word_path']
            with open(current_word_path, 'rb') as f:
                common_terms = pickle.load(f)
            test_trend = test_trend[common_terms]

        x_test, embedding_dim = get_feature_from_config(feature_pars, test_pol, test_phys, test_trend)

        model = get_model_from_config(feature_pars, model_type, embedding_dim)
        # build model
        model.build_model()
        model.load(feature_pars['save_model_path'])

        pred_class, pred_score = model.predict(x_test)
        result_scores = result_stat(y_test, pred_class, pred_score)
        print(result_scores)
        result_scores = [feature_pars['feature']] + result_scores
        record_pd = write_report(result_scores)

        report_pardir = os.path.dirname(report_path)
        if not os.path.exists(report_pardir):
            os.makedirs(report_pardir)
        record_pd.to_csv(report_path, index=False)
        # save config file
        save_config_path = os.path.join(report_pardir, 'config.ini')
        with open(save_config_path, 'w') as configfile:
            pars.write(configfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_file()
