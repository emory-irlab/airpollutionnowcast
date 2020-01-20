'''
    Author: Chen Lin
    Email: chen.lin@emory.edu
    Date created: 2020/1/20 
    Python Version: 3.6
'''

from keras import backend as K
import tensorflow as tf
import keras


# Compatible with tensorflow backend

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=.25,
             reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        gamma = self.gamma
        alpha = self.alpha
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) - \
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)



# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
#             (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#
#     return focal_loss_fixed
