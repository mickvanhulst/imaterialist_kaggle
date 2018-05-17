import numpy as np
from collections import Counter
import json
import pandas as pd


def _labels_to_array(labels, n_classes):
    labels_array = np.zeros(n_classes)
    labels_array[labels] = 1
    return labels_array

def __find_infreq_classes(df, label_occ_threshold):
    # Get labels to be ignored.
    total_list = []
    intermediary_var = df["labelId"].apply(lambda x: [total_list.append(i) for i in x])
    count_items = Counter(total_list)
    labels_whitelist = [x for x in count_items if count_items[x] > label_occ_threshold]

    # Remove labels that are in the blacklist
    df["labelId"] = df["labelId"].apply(lambda x: [i for i in x if i in labels_whitelist])
    return df

def __get_class_weights(series, n_classes):
    # Create one-hot encoding for the labels
    y = np.empty((len(series), n_classes), dtype=np.int)

    for i, item in series.iteritems():
        labels = np.asarray(item)

        # Store label and class
        y[i,] = _labels_to_array(labels, n_classes)

    # Count per column
    counter = y.sum(axis=0)

    # Calculate and return weights
    majority = np.max(counter)
    class_weights = {i: 0 if counter[i] == 0 else float(majority / counter[i]) for i in range(len(counter))}
    return class_weights

with open('../data/train.json', 'r') as f:
    train_data = json.load(f)

df = pd.DataFrame.from_records(train_data["images"])

train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
df = pd.merge(df, train_labels_df, on="imageId", how="outer")
df["labelId"] = df["labelId"].apply(lambda x: [int(i) for i in x])

df['imageId'] = df['imageId'].apply(lambda x: int(x))

df = __find_infreq_classes(df, 5000)
class_weights = __get_class_weights(df['labelId'], 228)




# Class weights
# https://stackoverflow.com/questions/43481490/keras-class-weights-class-weight-for-one-hot-encoding <-- third post
# https://datascience.stackexchange.com/questions/22170/setting-class-weights-for-categorical-labels-in-keras-using-generator <- voor als we op rij specifiek niveau willen checken.
#https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950 <-- try this666

# https://github.com/keras-team/keras/issues/2115