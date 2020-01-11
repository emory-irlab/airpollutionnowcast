import configparser
import logging
import sys

import click
import os

sys.path.append('.')
from src.evaluation.utils import process_features, result_stat, write_report, get_feature_array, get_rf_model,\
  get_two_branch_feature, get_lstm_model, get_feature_from_config, get_model_from_config


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('test_data_path', type=click.Path(exists=True))
def extract_file(config_path, test_data_path):
    logger = logging.getLogger(__name__)
    logger.info('predict the testing data')

    pars = configparser.ConfigParser()
    pars.read(config_path)

    seq_length = int(pars['train_model']['seq_length'])
    search_lag = int(pars['train_model']['search_lag'])
    # parameters for rf model
    model_path = pars['train_model']['save_model_path']
    model_type = pars['train_model']['model_type']
    # report path
    report_path = pars['predict_model']['report_path']

    y_test, test_pol, test_phys, test_trend = process_features(test_data_path, seq_length, search_lag)

    x_test, embedding_dim = get_feature_from_config(pars, model_type, test_pol, test_phys, test_trend, seq_length)
    model = get_model_from_config(pars, model_type, embedding_dim)

    # build model
    model.build_model()
    model.load(model_path)
    pred_class, pred_score = model.predict(x_test)
    result_scores = result_stat(y_test, pred_class, pred_score)
    print(result_scores)
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
