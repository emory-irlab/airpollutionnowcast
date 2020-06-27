'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/8 
    Python Version: 3.6
'''

import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score

# metrics to return f1
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

# function to return accuracy and f1
def result_stat(y_true, y_prediction, pred_score, test_nas_indices):
    # remove nas from stat
    y_true = np.array(y_true)[~test_nas_indices]
    y_prediction = np.array(y_prediction)[~test_nas_indices]
    pred_score = np.array(pred_score)[~test_nas_indices]

    cnf_matrix = confusion_matrix(y_true, y_prediction)

    # if cnf_matrix.shape[0] == 1:
    #     return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # true negatives, false positives, false negatives, true positives
    try:
        tn, fp, fn, tp = cnf_matrix.ravel()
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f1_value = f1_score(y_true, y_prediction)
    accuracy = accuracy_score(y_true, y_prediction)

    # auc_value
    fpr, tpr, threshold = roc_curve(y_true, pred_score)
    auc_value = auc(fpr, tpr)

    return accuracy, f1_value, tp, fp, tn, fn, auc_value
