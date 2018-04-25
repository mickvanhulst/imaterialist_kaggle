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

# %matplotlib inline

import os
# print(os.listdir("./input"))





def open_file(train=False, test=False, val=False):

    if(train == True):
        with open("../data/train.json") as datafile1: #first check if it's a valid json file or not
            train_data = json.load(datafile1)
    # if(test == True):
    #     with open("../input/test.json") as datafile2:  # first check if it's a valid json file or not
    #         test_data = json.load(datafile2)
    # if(val == True):
    #     with open("../input/validation.json") as datafile3:  # first check if it's a valid json file or not
    #         valid_data = json.load(datafile3)

    return train_data



if __name__ == "__main__":
    train_data = open_file(train=True)

    train_imgs_df = pd.DataFrame.from_records(train_data["images"])
    train_imgs_df["url"] = train_imgs_df["url"]
    train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
    # train_labels_df = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
    train_df = pd.merge(train_imgs_df, train_labels_df, on="imageId", how="outer")
    train_df["imageId"] = train_df["imageId"].astype(np.int)
    print(train_df.head())
    print(train_df.dtypes)

    print("####" * 10)
    print("## Training Data.")
    print(train_df.isna().any())

    train_image_arr = train_df[["imageId", "labelId"]].apply(lambda x: [(x["imageId"], int(i)) for i in x["labelId"]],
                                                             axis=1).tolist()
    train_image_arr = [item for sublist in train_image_arr for item in sublist]
    train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
    train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)
    train_image_vals = np.ones(len(train_image_col))



    train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))


    labels = train_image_mat.sum(0).astype(np.int)
    # print(labels)

    plt.figure(figsize=(30, 18))
    labels_inds = np.arange(len(labels.tolist()[0]))
    sns.barplot(labels_inds, labels.tolist()[0])
    plt.xlabel('label id', fontsize=6)
    plt.ylabel('Count', fontsize=16)
    plt.title("Distribution of labels", fontsize=18)
    plt.show()



    ## Rows = images  //  columns classes
    df_train_image_mat = pd.DataFrame(train_image_mat.toarray())

    print(df_train_image_mat.iloc[2:5])
    print("train image mat als df", list(df_train_image_mat))


