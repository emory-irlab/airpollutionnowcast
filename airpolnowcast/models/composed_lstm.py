'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2019/9/21
    Python Version: 3.6
'''
import sys
sys.path.append('.')
from airpolnowcast.models.composed_model import ComposedModel
from airpolnowcast.models.lstm import LSTMModel


class ComposedLSTM(ComposedModel, LSTMModel):
    pass
