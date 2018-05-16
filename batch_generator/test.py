import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import class_weight

with open('../data/train.json', 'r') as f:
    train_data = json.load(f)

train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
train_labels_df["labelId"] = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])

# Get labels to be ignored.
total_list = []
intermediary_var = train_labels_df["labelId"].apply(lambda x: [total_list.append(i) for i in x])
count_items = Counter(total_list)
labels_blacklist = [x for x in count_items if count_items[x] <= 5000]


# Class weights
# https://stackoverflow.com/questions/43481490/keras-class-weights-class-weight-for-one-hot-encoding <-- third post
# https://datascience.stackexchange.com/questions/22170/setting-class-weights-for-categorical-labels-in-keras-using-generator <- voor als we op rij specifiek niveau willen checken.
#https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950 <-- try this666
