from functools import partial

from keras import backend as K


def get_loss(weights):
    weights = K.variable(weights)
    return partial(weighted_categorical_crossentropy, weights=weights)


def weighted_categorical_crossentropy(y_true, y_pred, weights):
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss
