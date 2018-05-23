from functools import partial, update_wrapper

from keras import backend as K
import tensorflow as tf

import numpy as np

weights = np.ones((228,))


def weighted_categorical_crossentropy(class_weights):
    global weights
    weights = K.variable(class_weights)

    return _weighted_categorical_crossentropy


def _weighted_categorical_crossentropy(y_true, y_pred):
    _loss = y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)
    _loss = tf.multiply(_loss, weights)
    _loss = -K.sum(_loss, -1)
    return _loss


def weighted_mean_squared_error(class_weights):
    global weights
    weights = K.variable(class_weights)

    return _weighted_mean_squared_error


def _weighted_mean_squared_error(y_true, y_pred):
    _loss = K.mean(K.square(y_pred - y_true) * weights, axis=-1)
    return _loss
