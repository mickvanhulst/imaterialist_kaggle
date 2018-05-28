import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.inceptionv3 import inception_v3_model
from networks.xception import xception_model
from utils.load_model import load_model

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
        y[i, ] = labels_to_array(labels, n_classes)

    return y


def thresholding(model_name, model_class, datafile='../data/validation.json', data_path='../data/img/validation/'):
    y_true = get_labels()

    val_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    val_generator = val_generator.make_datagenerator(
        datafile=datafile, data_path=data_path, save_images=True,
        test=False, shuffle=False, batch_size=64)

    model = load_model(model_name)

    y_pred = model.predict_generator(val_generator,
                                     steps=None,
                                     verbose=1)

    np.save("y_pred", y_pred)

    # y_pred = np.load('./y_pred.npy')

    thresholds = np.linspace(0.0, 1.0, num=50)
    best_thresh_list = []
    for n_class in tqdm(range(228), desc="Calculating Thresholds", unit="classes"):
        y_true_temp = y_true[:, n_class]
        y_pred_temp = y_pred[:, n_class]

        high_score = 0
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred_thresh = y_pred_temp.copy()

            y_pred_thresh[y_pred_thresh >= threshold] = 1
            y_pred_thresh[y_pred_thresh < threshold] = 0

            # Compare prediction with true labels.
            score = f1_score(y_true_temp, y_pred_thresh, average="micro")
            score = 0 if np.isnan(score) else score

            # print("f1 score", score)

            if score > high_score:
                best_threshold = threshold
                high_score = score
            # print("high score ", high_score, "threshold ", best_threshold)
        best_thresh_list.append(best_threshold)

    for i, val in enumerate(y_pred):
        val[val >= best_thresh_list] = 1
        val[val < best_thresh_list] = 0

    print(best_thresh_list)
    np.savetxt("thresholds_{}.npy".format(model_name), best_thresh_list)

    return best_thresh_list, y_pred


if __name__ == "__main__":
    model_name = "../trainer/model_epoch09.h5"
    model_class = xception_model
    thresholding(model_name, model_class)
