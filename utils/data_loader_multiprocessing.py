import io
import json
import multiprocessing
import os

import pandas as pd
import urllib3
from PIL import Image
from tqdm import tqdm

import numpy as np
from urllib3 import Retry

import matplotlib.pyplot as plt

urllib3.disable_warnings()
tqdm.pandas()

http_pool = urllib3.PoolManager(500, retries=Retry(connect=3, read=2, redirect=3))


def pad_image(img, final_size=(299, 299), pad_color=(255, 255, 255)):
    x, y = img.size

    # if (x < final_size[0]) and (y < final_size[1]):
    #     new_img = Image.new("RGB", final_size, color=pad_color)
    #     new_img.paste(img, ((final_size[0] - x) // 2, (final_size[1] - y) // 2))
    if (x == final_size[0]) and (y == final_size[1]):
        new_img = img
    else:
        new_img = Image.new("RGB", final_size, color=pad_color)
        larger_dim = 0 if x > y else 1
        if larger_dim == 0:
            new_x = final_size[0]
            new_y = int(y * new_x / x)
        else:
            new_y = final_size[1]
            new_x = int(x * new_y / y)
        img_resize = img.resize((new_x, new_y), Image.ANTIALIAS)
        new_img.paste(img_resize, ((final_size[0] - new_x) // 2, (final_size[1] - new_y) // 2))

    return new_img


def download_image(pair):
    """
    Download an images given its URL and save the normalized version of it
    :param pair: [url to the image, name of the image]
    :return:
    """
    data_dir = "../data/img_big/train/"
    # resize_dim = (224, 224)

    url = pair[0]
    ID = pair[1]

    # check if the image is already there
    save_path = os.path.join(data_dir, str(ID) + ".jpg")
    if os.path.isfile(save_path):
        try:
            Image.open(save_path)
            return
        except Exception:
            os.remove(save_path)

    try:
        response = http_pool.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image = image.convert("RGB")
        if np.min(np.array(image)) > 180:  # Don't save images containing nothing
           return
        image = pad_image(image)
        image.save(save_path, optimize=True)
    except Exception as e:
        print("Image '{}' could not be loaded from '{}'.\n{}".format(ID, url, e))


def download_data(df_path):
    """
    Downloads all the images in the df to the data directory.
    :param df_path: path to the json file
    :return:
    """
    with open(df_path, 'r') as f:
        data = json.load(f)

    imgs_df = pd.DataFrame.from_records(data["images"])
    del data

    n_samples = len(imgs_df)
    pairs = zip(imgs_df.values[:, 1], imgs_df.values[:, 0])
    del imgs_df

    pool = multiprocessing.Pool(processes=12)  # How much can you handle tho?

    with tqdm(total=n_samples, desc="Processing Dataset", unit="images") as pbar:
        for _ in pool.imap_unordered(download_image, pairs):
            pbar.update(1)


if __name__ == "__main__":
    data_dir = "../data/img_big/train/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download_data("../data/train.json")
