from functools import partial, update_wrapper

from keras import backend as K
import tensorflow as tf


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        _loss = y_true * K.log(y_pred)
        _loss = tf.multiply(_loss, weights)
        _loss = -K.sum(_loss, -1)
        return _loss

    return loss


def actual_correct(y_true, y_pred):
    loss = tf.multiply(y_true, y_pred)
    loss = 100 / K.sum(loss, -1)
    return loss
