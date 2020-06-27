'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/2/5 
    Python Version: 3.6
'''

import sys
sys.path.append('.')
from airpolnowcast.models.dict_learner import DLLSTMModel

import datetime

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, LSTM
from tensorflow.keras.models import Model
from airpolnowcast.models.embedding_utils import get_glove_and_intent

n_tasks = 10


class MTDLLSTM(DLLSTMModel):

    def build_sensor_branch(self, sensor_input_list):

        # search_input = Input(shape=(self.seq_length, self.embedding_dim))
        new_embedding_dim = 100
        return_seq = False

        # lstm = LSTM(neuron_num, dropout=0.5, recurrent_dropout=0.5, kernel_initializer=he_normal(seed=1))
        outs = []

        for i, input_i in enumerate(sensor_input_list[:10]):
            glove_embedding_input = sensor_input_list[i]
            search_input = sensor_input_list[n_tasks+i]
            # Transform search interest data to incorporate term embeddings.
            new_embedding = Dense(new_embedding_dim, kernel_initializer=he_normal(seed=11), activation='relu')(
                glove_embedding_input[0])
            batch_seq = tf.reshape(search_input, [-1, self.embedding_dim])

            batch_new_seq = tf.matmul(batch_seq, new_embedding)

            batch_new_seq = Dense(new_embedding_dim, kernel_initializer=he_normal(seed=11), activation='relu')(
                batch_new_seq)
            new_search_seq = tf.reshape(batch_new_seq, [-1, self.seq_length, new_embedding_dim])

            net = self.shared_module(new_search_seq, return_seq)
            outs.append(net)

        dense = Dense(64, activation='relu', kernel_initializer=he_normal(seed=1))
        outs_2 = []
        for i, out in enumerate(outs):
            outs_2.append(dense(out))

        dropout = Dropout(0.5)
        outs_3 = []
        for i, out in enumerate(outs_2):
            outs_3.append(dropout(out))
        return outs_3

    def build(self):

        # neuron_num = 128
        word_embedding_dim = 50 + 100
        n_words = 51

        # Model inputs:

        sensor_input_list = [Input(shape=(n_words, word_embedding_dim)) for _ in range(n_tasks)] + [Input(shape=(self.seq_length, self.embedding_dim)) for _ in range(n_tasks)]

        sensor_output_list = self.build_sensor_branch(sensor_input_list)

        outputs = []
        for i in range(n_tasks):
            task_i_predictors = sensor_output_list[i]
            task_i_predictions = Dense(1, activation='sigmoid', kernel_initializer=he_normal(seed=1))(task_i_predictors)
            outputs.append(task_i_predictions)

        model = Model(inputs=sensor_input_list, outputs=outputs)
        return model

    def fit(self, x_train, x_valid, y_train, y_valid):

        l = y_train.shape[0]
        l2 = y_valid.shape[0]

        def make_tl(x):
            l = x.shape[0]
            return [x[int(i):int(i + l / n_tasks)] for i in range(0, int(l / n_tasks) * n_tasks, int(l / n_tasks))]

        x_train_list = make_tl(x_train)
        x_valid_list = make_tl(x_valid)
        y_train_list = [y_train[int(i):int(i + l / n_tasks)] for i in range(0, int(l / n_tasks) * n_tasks, int(l / n_tasks))]
        y_valid_list = [y_valid[int(i):int(i + l2 / n_tasks)] for i in
                        range(0, int(l2 / n_tasks) * n_tasks, int(l2 / n_tasks))]

        glove_embedding = get_glove_and_intent(self.filtered_dict_path, self.intent_dict_path, self.current_word_path)
        glove_embedding_tr = np.tile(glove_embedding, (int(l / n_tasks), 1, 1))
        glove_embedding_vl = np.tile(glove_embedding, (int(l2 / n_tasks), 1, 1))
        glove_embedding_trvl = np.tile(glove_embedding, (int(l / n_tasks) + int(l2 / n_tasks), 1, 1))

        # Rest of function assumes input is in tuple of lists format:
        # ([..., task_i_search_input, ...], [...,task_i_sensor_input,...])
        def arr_concate(train_arr, valid_arr):
            return np.concatenate([train_arr, valid_arr], axis=0)

        x_train_valid_list = []
        y_train_valid_list = []
        for i in range(len(x_train_list)):
            x_train_valid_list.append(arr_concate(x_train_list[i], x_valid_list[i]))
            y_train_valid_list.append(arr_concate(y_train_list[i], y_valid_list[i]))

        print(glove_embedding.shape, glove_embedding_tr.shape, glove_embedding_vl.shape, glove_embedding_trvl.shape)

        # first shape
        print(np.array(x_train_list).shape)
        # print(x_train_list)

        x_train_list = [glove_embedding_tr for _ in range(n_tasks)] + x_train_list
        x_valid_list = [glove_embedding_vl for _ in range(n_tasks)] + x_valid_list
        x_train_valid_list = [glove_embedding_trvl for _ in range(n_tasks)] + x_train_valid_list

        # print(np.array(x_train_list).shape)

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=self.patience)

        logdir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb = TensorBoard(log_dir=logdir)

        # save initial weights
        tmp_model_path = os.path.join('models/interim', str(os.getpid()) + 'model.h5')
        self.model.save_weights(tmp_model_path)

        # calculate the weights
        class_weights = []
        for y in y_train_valid_list:
            (_, data_count) = np.unique(y, return_counts=True)
            class_weights.append({0: sum(data_count) / data_count[0], 1: sum(data_count) / data_count[1]})

        class_weight = {k: class_weights[i] for i, k in enumerate(
            ['dense_1', 'dense_2', 'dense_3', 'dense_4', 'dense_5', 'dense_6', 'dense_7', 'dense_8', 'dense_9',
             'dense_10'])}
        # print(class_weight)
        max_epochs = 1000
        min_epochs = 15

        history = self.model.fit(x_train_list, y_train_list, batch_size=self.batch_size,
                                 epochs=max_epochs,
                                 validation_data=(x_valid_list, y_valid_list),
                                 class_weight=class_weight,
                                 verbose=1,
                                 callbacks=[tb, es], shuffle=True)

        epochs = max(len(history.epoch) - self.patience, min_epochs)
        # restore initial weights
        self.model.load_weights(tmp_model_path)

        self.model.fit(x_train_valid_list, y_train_valid_list,
                       batch_size=self.batch_size,
                       epochs=epochs, class_weight=class_weight,
                       verbose=1)
        # remove model file
        os.remove(tmp_model_path)

    def predict(self, x_test):
        def make_tl(x):
            l = x.shape[0]
            return [x[int(i):int(i + l / n_tasks)] for i in range(0, int(l / n_tasks) * n_tasks, int(l / n_tasks))]

        x_test_list = make_tl(x_test)
        pred_score_list = self.model.predict(x_test_list)
        pred_class_list = [[0 if i < 0.5 else 1 for i in pred_score] for pred_score in pred_score_list]

        return pred_class_list, pred_score_list
