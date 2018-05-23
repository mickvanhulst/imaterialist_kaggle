import keras
import networks.loss as loss
from keras.utils.generic_utils import CustomObjectScope
import os


def load_model(path):
    if os.path.isfile(path):
        with CustomObjectScope(
                {'relu6': keras.applications.mobilenet.relu6,
                 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
                 '_weighted_categorical_crossentropy': loss._weighted_categorical_crossentropy,
                 '_weighted_mean_squared_error': loss._weighted_mean_squared_error}):
            return keras.models.load_model(path)
    else:
        raise Exception("Model {} not found!".format(path))
