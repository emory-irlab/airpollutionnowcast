import sys
sys.path.append('.')
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, Activation, LSTM, GRU, Reshape, Dropout

from src.models.model import Model


class LSTMModel(Model):

    def shared_module(self, input1, return_seq=False):
        neuron_num = 128
        net = LSTM(neuron_num, dropout=0.5, recurrent_dropout=0.5, kernel_initializer=he_normal(seed=1), return_sequences=return_seq)(input1)

        if return_seq:
            net = Reshape((7*neuron_num, ), input_shape=(7, neuron_num))(net)

        net = Dense(64, activation='relu', kernel_initializer=he_normal(seed=1))(net)

        net = Dropout(0.5)(net)

        return net


