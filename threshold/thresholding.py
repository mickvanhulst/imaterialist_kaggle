from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.mobilenet import mobilenet_model
from networks.inceptionv3 import inception_v3_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path
import networks.loss as loss
import json
from sklearn.metrics import f1_score
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

def thresholding(model_name, model_class):
    y_true = get_labels()


    val_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    val_generator = val_generator.make_datagenerator(
        datafile='../data/validation.json', data_path='../data/img/val/', save_images=False,
        test=False, shuffle=False, batch_size=128)

    model = load_model("../incept_all_model.h5")

    # y_pred = model.predict_generator(val_generator,
    #                                  steps=None,
    #                                  verbose=1)

    # np.save("y_pred", y_pred)

    y_pred = np.load('./y_pred.npy')

    thresholds = np.linspace(0.0, 1.0, num=50)
    best_thresh_list = []
    for n_class in range(228):
        y_true_temp = y_true[:, n_class]
        y_pred_temp = y_pred[:, n_class]

        high_score = 0
        for threshold in thresholds:
            y_pred_thresh = y_pred_temp.copy()

            y_pred_thresh[y_pred_thresh >= threshold] = 1
            y_pred_thresh[y_pred_thresh < threshold] = 0

            # Compare prediction with true labels.
            score = (y_true_temp == y_pred_thresh).sum()

            # TODO: Want to use F1score above, but gives error about 0 divide error

            # print("f1 score", score)

            if score > high_score:
                best_threshold = threshold
                high_score = score
            # print("high score ", high_score, "threshold ", best_threshold)
        best_thresh_list.append(best_threshold)

    #np.argwhere(pred > best_thresh_list).flatten()
    for  i, val in enumerate(y_pred):
        y_pred[y_pred[i] >= best_thresh_list] = 1
        y_pred[y_pred[i] < best_thresh_list] = 0

    # print((best_thresh_list))
    # print(np.unique(best_thresh_list))
    print(best_thresh_list)



    np.savetxt("thresholds_incept_all_model",best_thresh_list)




if __name__ == "__main__":
    model_name = "incept_all_model.h5"
    model_class = inception_v3_model
    thresholding(model_name,model_class)
