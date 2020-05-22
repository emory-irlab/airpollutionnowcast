'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/5/21 
    Python Version: 3.6
'''
import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model as keras_Model

from src.models.embedding_utils import get_glove_and_intent, get_wv_dict
from src.models.lstm import LSTMModel


class DLLSTMModel(LSTMModel):

    def build(self):
        word_embedding_dim = 50 + 100
        hidden_dim = 100
        n_words = 152
        return_seq = False

        # Model inputs.
        glove_embedding_input = Input(shape=(n_words, word_embedding_dim))
        search_input = Input(shape=(self.seq_length, self.embedding_dim))

        # Transform search interest data to incorporate term embeddings.
        new_embedding = glove_embedding_input[0]
        batch_seq = tf.reshape(search_input, [-1, self.embedding_dim])
        batch_new_seq = tf.matmul(batch_seq, new_embedding)
        batch_new_seq = Dense(hidden_dim, kernel_initializer=he_normal(seed=11), activation='relu')(batch_new_seq)
        new_search_seq = tf.reshape(batch_new_seq, [-1, self.seq_length, hidden_dim])

        net = self.shared_module(new_search_seq, return_seq)
        out = Dense(1, activation='sigmoid', kernel_initializer=he_normal(seed=1))(net)
        model = keras_Model(inputs=[glove_embedding_input, search_input], outputs=out)

        return model

    def get_glove_and_intent_path(self, pars):
        self.intent_dict_path = pars['intent_dict_path']
        self.filtered_dict_path = pars['filtered_dict_path']
        self.current_word_path = pars['current_word_path']

    def fit(self, x_train, x_valid, y_train, y_valid):

        glove_embedding = get_wv_dict(self.filtered_dict_path, self.current_word_path)
        glove_embedding_tr = np.tile(glove_embedding, (x_train.shape[0], 1, 1))
        glove_embedding_vl = np.tile(glove_embedding, (x_valid.shape[0], 1, 1))
        glove_embedding_trvl = np.tile(glove_embedding, (x_valid.shape[0] + x_train.shape[0],1 , 1))

        def arr_concate(train_arr, valid_arr):
            return np.concatenate([train_arr, valid_arr], axis=0)

        # concatenate the train and valid inputs
        x_train_valid = arr_concate(x_train, x_valid)

        y_train_valid = arr_concate(y_train, y_valid)

        print(glove_embedding.shape, glove_embedding_tr.shape, glove_embedding_vl.shape, glove_embedding_trvl.shape,
              x_train_valid.shape, y_train_valid.shape)
        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                           patience=self.patience)

        logdir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb = TensorBoard(log_dir=logdir, histogram_freq=10)

        # save initial weights
        tmp_model_path = os.path.join('models/interim', str(os.getpid()) + 'model.h5')
        self.model.save_weights(tmp_model_path)

        # calculate class weights
        (_, data_count) = np.unique(y_train_valid, return_counts=True)
        class_weight = {0: sum(data_count)/data_count[0], 1: sum(data_count)/data_count[1]}

        max_epochs = 1000
        min_epochs = 10
        callbacks = [tb, es]
        epochs = max_epochs
        if es in callbacks:
            history = self.model.fit([glove_embedding_tr, x_train], y_train, batch_size=self.batch_size,
                                     epochs=max_epochs, validation_data=((glove_embedding_vl,x_valid), y_valid),
                                     class_weight=class_weight, verbose=1,
                                     callbacks=callbacks, shuffle=True)
            epochs = max(len(history.epoch) - self.patience, min_epochs)
            # restore initial weights
            self.model.load_weights(tmp_model_path)
            # remove model file
            os.remove(tmp_model_path)
            callbacks = []

        self.model.fit([glove_embedding_trvl, x_train_valid], y_train_valid,
                       batch_size=self.batch_size, callbacks=callbacks,
                       epochs=epochs, class_weight=class_weight, verbose=1)

    def predict(self, x_test):
        glove_embedding = get_glove_and_intent(self.filtered_dict_path, self.intent_dict_path, self.current_word_path)
        glove_embedding_ts = np.tile(glove_embedding, (x_test.shape[0],1 , 1))
        pred_score = self.model.predict((glove_embedding_ts, x_test))
        pred_class = [0 if i < 0.5 else 1 for i in pred_score]

        return pred_class, pred_score