from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.mobilenet import mobilenet_model
from networks.inceptionv3 import inception_v3_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path
import networks.loss as loss
import json

import matplotlib.pyplot as plt
import numpy as np

from utils import params
from utils.load_model import load_model

from keras import optimizers
import pandas as pd

'''
Idee:
1. Per model alles predicten op validatieset
2. Weights trainen voor ensemblen per resultaat validatieset. Hier gebruiken we een sklearn wrapper voor.
3. Na het trainen van de weights weten we hoe vaak predictions meetellen. Die gebruiken we dan om de thresholds te optimaliseren (met de F1 score) op de validatieset. Zodra we de threshold per class hebben zijn we klaar.
4. Voorspel test set, apply ensemble learning + weights en thresholds.
'''


def labels_to_array(labels, n_classes):
    """
    Converts a list of labels to a one-hot encoded array
    """
    labels_array = np.zeros((n_classes,), dtype=int)
    labels = np.array(labels)

    # Labels are 1-based, so do - 1
    if len(labels) > 0:
        labels_array[labels - 1] = 1

    return labels_array

def get_labels():
    # Load data
    datafile = '../data/validation.json'
    with open(datafile, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_records(data["images"])
    labels_df = pd.DataFrame.from_records(data["annotations"])
    df = pd.merge(df, labels_df, on="imageId", how="outer")
    df["labelId"] = df["labelId"].apply(lambda x: [int(i) for i in x])
    df['imageId'] = df['imageId'].astype(int, copy=False)
    series = df["labelId"]
    n_classes = 228

    # Get matrix y
    y = np.empty((len(series), n_classes), dtype=np.int)

    for i, item in series.iteritems():
        labels = np.asarray(item)

        # Store label and class
        y[i,] = labels_to_array(labels, n_classes)

    return y

def thresholding(model_name,model_class):
    y = get_labels()
    print(y)

    # val_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    # val_generator = val_generator.make_datagenerator(
    #     datafile='./data/validation.json', data_path='./data/img/val/', save_images=False,
    #     test=False, shuffle=False, batch_size=128)
    #
    # model = load_model("./incept_all_model.h5")



    #batch_y = val_generator.

    #print(len(batch_y))

    # y_pred = model.predict_generator(val_generator,
    #                                 steps=None,
    #                                 verbose=1)

    #print(len(batch_y))
    #print(len(y_pred))



    return None



if __name__ == "__main__":
    model_name = "incept_all_model.h5"
    model_class = inception_v3_model
    thresholding(model_name,model_class)
