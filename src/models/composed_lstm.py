'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2019/9/21
    Python Version: 3.6
'''
import sys
sys.path.append('.')
from src.models.composed_model import ComposedModel
from src.models.lstm import LSTMModel


class ComposedLSTM(ComposedModel, LSTMModel):
    pass
