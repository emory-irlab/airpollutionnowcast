'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/8 
    Python Version: 3.6
'''

import sys

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model as keras_Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TruePositives, TrueNegatives, Recall
import datetime
sys.path.append('.')
from src.models.metrics import f1


class Model(object):
    def __init__(self, seq_length, embedding_dim, learning_rate=0.001, batch_size=32, patience=15, log_dir=None):
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.log_dir = log_dir
        print(self.patience, self.learning_rate)

    def build_model(self):
        with tf.name_scope('Model'):
            self.model = self.build()
            self.model.compile(optimizer=Adam(lr=self.learning_rate),
                               loss='binary_crossentropy',
                               metrics=['accuracy', f1, TrueNegatives(),
                                        FalseNegatives(), TruePositives(),
                                        FalsePositives()])
        print(self.model.summary())

    def build(self):

        reg = None
        # model structure
        input1 = Input(shape=(self.seq_length, self.embedding_dim))
        first_dense = self.shared_module(input1)

        output = Dense(1, activation='sigmoid',
                       activity_regularizer=reg,
                       kernel_initializer=he_normal(seed=1))(first_dense)

        model = keras_Model(inputs=input1, outputs=output)
        return model

    def load(self, fname):
        self.model.load_weights(fname)

    def fit(self, x_train, x_valid, y_train, y_valid):
        def arr_concate(train_arr, valid_arr):
            return np.concatenate([train_arr, valid_arr], axis=0)

        # concatenate the trian and valid inputs
        if isinstance(x_train, tuple):
            x_train_valid = [arr_concate(x_train[0], x_valid[0]), arr_concate(x_train[1], x_valid[1])]
        else:
            x_train_valid = arr_concate(x_train, x_valid)

        y_train_valid = arr_concate(y_train, y_valid)

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                           patience=self.patience)

        logdir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb = TensorBoard(log_dir=logdir)
        # save initial weights
        tmp_model_path = os.path.join('models/interim', str(os.getpid()) + 'model.h5')
        self.model.save_weights(tmp_model_path)

        # calculate the weights
        (_, data_count) = np.unique(y_train_valid, return_counts=True)
        class_weight = {0: float(data_count[1]), 1: float(data_count[0])}

        history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                 epochs=1000, validation_data=(x_valid, y_valid),
                                 class_weight=class_weight, verbose=1,
                                 callbacks=[tb, es], shuffle=True)

        best_epoch = max(len(history.epoch) - self.patience, 10)

        # restore initial weights
        self.model.load_weights(tmp_model_path)

        self.model.fit(x_train_valid, y_train_valid,
                       batch_size=self.batch_size,
                       epochs=best_epoch, class_weight=class_weight, verbose=1, shuffle=True)
        # remove model file
        os.remove(tmp_model_path)

    def predict(self, x_test):
        pred_score = self.model.predict(x_test)
        pred_class = [0 if i < 0.5 else 1 for i in pred_score]

        return pred_class, pred_score

    def save(self, fname):
        self.model.save_weights(fname)
