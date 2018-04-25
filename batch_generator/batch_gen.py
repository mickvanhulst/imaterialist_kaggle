import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame
import gc
from IPython.display import Image
from IPython.core.display import HTML

from scipy.sparse import csr_matrix

def open_file(train=False, test=False, val=False):

    if(train == True):
        with open("../data/train.json") as datafile1: #first check if it's a valid json file or not
            train_data = json.load(datafile1)

    return train_data


def prob_scores(df):
    # Count
    agg_classes = train_df.groupby(['label_str']).size().reset_index().rename(columns={0:'cnt'})
    return  1 - (agg_classes['cnt'] / len(train_df))



if __name__ == "__main__":
    train_data = open_file(train=True)

    train_imgs_df = pd.DataFrame.from_records(train_data["images"])
    train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
    # train_labels_df = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
    train_df = pd.merge(train_imgs_df, train_labels_df, on="imageId", how="outer")
    train_df["imageId"] = train_df["imageId"].astype(np.int)
    train_df['label_str'] = train_df['labelId'].apply(lambda x: ' '.join(x))

    # TODO: for task 1 + 2
    #train_df['probs'] = prob_scores(train_df)

    '''

    train_df['probs'] = pd.merge(train_df, agg_classes[['label_str', 'probs']], how='left', on='label_str')

    # Keras sequence to cat using apply
    train_image_arr = train_df[["imageId", "labelId"]].apply(lambda x: [(x["imageId"], int(i)) for i in x["labelId"]],
                                                             axis=1).tolist()
    train_image_arr = [item for sublist in train_image_arr for item in sublist]
    train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
    train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)
    train_image_vals = np.ones(len(train_image_col))



    train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))


    labels = train_image_mat.sum(0).astype(np.int)


    ## Rows = images  //  columns classes
    df_train_image_mat = pd.DataFrame(train_image_mat.toarray())
    '''

