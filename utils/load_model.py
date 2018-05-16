import keras
from keras.utils.generic_utils import CustomObjectScope
import os


def load_model(path):
    if os.path.isfile(path):
        if "mobilenet" in path:
            with CustomObjectScope(
                    {'relu6': keras.applications.mobilenet.relu6,
                     'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                return keras.models.load_model(path)
        else:
            return keras.models.load_model(path)
    else:
        raise Exception("Model {} not found!".format(path))
