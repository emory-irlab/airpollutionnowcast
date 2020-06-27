'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/5/20 
    Python Version: 3.6
'''

import unittest

from configparser import ExtendedInterpolation
import configparser
import argparse
import sys
import os
sys.path.append(os.getcwd())
from airpolnowcast.features.build_features import FeatureEngineer
from airpolnowcast.data.utils import read_query_from_file
from airpolnowcast.evaluation.utils import get_feature_pars
import ast


class TestBuildFeatures(unittest.TestCase, object):

    def setUp(self):
        self.feature_engineer = FeatureEngineer()
        self.pars = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        self.pars.read(config_path)
        seq_length = int(self.pars['train_model']['seq_length'])
        search_lag = int(self.pars['train_model']['search_lag'])
        seed_path = self.pars['extract_search_trend']['term_list_path']
        seed_word_list = read_query_from_file(seed_path)
        self.y_train, self.train_pol, self.train_phys, train_trend = \
            self.feature_engineer.feature_from_file(train_data_path, seq_length, search_lag)
        self.train_trend = train_trend[seed_word_list]
        use_feature = ast.literal_eval(self.pars['train_model']['use_feature'])
        self.test_1_index = use_feature[0]
        self.test_2_index = use_feature[1]
        self.test_3_index = use_feature[2]

    def test_0(self):
        unit_y_train = ast.literal_eval(self.pars['unit_test_0']['unit_y_train'])
        self.assertListEqual(list(self.y_train), unit_y_train)

    def test_1(self):
        unit_embedding_dim = int(self.pars['unit_test_1']['unit_embedding_dim'])
        unit_x_train = ast.literal_eval(self.pars['unit_test_1']['unit_x_train'])
        feature_pars = get_feature_pars(self.pars, self.test_1_index)
        x_train, embedding_dim = self.feature_engineer.create_feature_sequence(feature_pars, self.train_pol,
                                    self.train_phys, self.train_trend)

        self.assertEqual(embedding_dim, unit_embedding_dim)
        self.assertListEqual(x_train[-2].tolist(), unit_x_train)
        self.assertListEqual(x_train[-3].tolist(), unit_x_train)
        self.assertListEqual(x_train[-4].tolist(), unit_x_train)
        # self.assertTrue(True)

    def test_2(self):
        unit_embedding_dim = ast.literal_eval(self.pars['unit_test_2']['unit_embedding_dim'])
        unit_pol_phys_branch_first_day = ast.literal_eval(self.pars['unit_test_2']['unit_pol_phys_branch_first_day'])
        unit_pol_phys_branch_last_day = ast.literal_eval(self.pars['unit_test_2']['unit_pol_phys_branch_last_day'])
        unit_trend_branch = ast.literal_eval(self.pars['unit_test_2']['unit_trend_branch'])
        feature_pars = get_feature_pars(self.pars, self.test_2_index)
        x_train, embedding_dim = self.feature_engineer.create_feature_sequence(feature_pars, self.train_pol,
                                    self.train_phys, self.train_trend)
        pol_phys_branch = x_train[0]
        trend_branch = x_train[1]
        self.assertListEqual(list(embedding_dim), unit_embedding_dim)

        self.assertListEqual(pol_phys_branch[0][-1].tolist(), unit_pol_phys_branch_first_day)
        self.assertListEqual(pol_phys_branch[-1][-1].tolist(), unit_pol_phys_branch_last_day)
        self.assertListEqual(trend_branch[-2].tolist(), unit_trend_branch)
        self.assertListEqual(trend_branch[-3].tolist(), unit_trend_branch)
        self.assertListEqual(trend_branch[-4].tolist(), unit_trend_branch)


if __name__ == '__main__':
    # sys.path.append(os.getcwd())
    # print(sys.path)
    config_path = sys.argv[2]
    train_data_path = sys.argv[3]
    del sys.argv[2:]

    # TestBuildFeatures.read_command_line(args.config_path, args.train_data_path)
    unittest.main()

