import io
import json
import keras.backend as K
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

import os

urllib3.disable_warnings()
tqdm.pandas()


def download_image(url, ID, data_dir, resize_dim):
    """
    Download an images given its URL and save the normalized version of it
    :param url: url to the image
    :param ID: name of the image
    :param data_dir: directory where the image should be stored
    :param resize_dim: dimension of the resulting image
    :return:
    """

    # check if the image is already there
    save_path = os.path.join(data_dir, str(ID))
    if os.path.isfile(save_path):
        return

    http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
    response = http.request("GET", url)
    image = Image.open(io.BytesIO(response.data))
    image = image.convert("RGB")
    image = image.resize(resize_dim)
    image = np.asarray(image, dtype=K.floatx())
    image = image / 255

    np.save(save_path, image)


def download_data(df_path, data_dir, resize_dim):
    """
    Downloads all the images in the df to the data directory.
    :param df_path: path to the json file
    :param data_dir: path to the directory where the images should be stored
    :param resize_dim: dimension of the resulting images
    :return:
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(df_path, 'r') as f:
        data = json.load(f)
    imgs_df = pd.DataFrame.from_records(data["images"])
    del data

    imgs_df.progress_apply(lambda row: download_image(row['url'], row['imageId'], data_dir, resize_dim), axis=1)


if __name__ == "__main__":
    download_data("../data/train.json", "../data/train/", resize_dim=(224, 224))

