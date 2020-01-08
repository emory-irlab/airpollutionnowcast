'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/8 
    Python Version: 3.6
'''

import sys
sys.path.append('.')
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model as keras_Model

from src.models.model import Model


class ComposedModel(Model):

    def build(self):
        # parameters of CNN network
        self.embedding_dim_1, self.embedding_dim_2 = self.embedding_dim

        # model structure
        # first input layer
        input1 = Input(shape=(self.seq_length, self.embedding_dim_1))
        first_dense = self.shared_module(input1)

        # second input layer
        input2 = Input(shape=(self.seq_length, self.embedding_dim_2))
        second_dense = self.shared_module(input2)

        # concatenate layers
        merged_layer = concatenate([first_dense, second_dense])

        output = Dense(1, activation='sigmoid', kernel_initializer=he_normal(seed=1))(
            merged_layer)

        model = keras_Model(inputs=[input1, input2], outputs=output)
        return model
