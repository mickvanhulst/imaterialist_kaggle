from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.mobilenet import mobilenet_model
from networks.inceptionv3 import inception_v3_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path
import networks.loss as loss

import matplotlib.pyplot as plt
import numpy as np

from utils import params
from utils.load_model import load_model

from keras import optimizers


def thresholding(model_name,model_class):

    val_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    val_generator = val_generator.make_datagenerator(
        datafile='./data/validation.json', data_path='./data/img/val/', save_images=False,
        test=False, shuffle=False, batch_size=128)

    model = load_model("./incept_all_model.h5")



    batch_y = val_generator.

    print(len(batch_y))

    # y_pred = model.predict_generator(val_generator,
    #                                 steps=None,
    #                                 verbose=1)

    print(len(batch_y))
    print(len(y_pred))



    return None



if __name__ == "__main__":
    model_name = "incept_all_model.h5"
    model_class = inception_v3_model
    thresholding(model_name,model_class)










