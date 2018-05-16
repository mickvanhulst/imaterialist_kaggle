import io
import json
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from tqdm import tqdm
import multiprocessing

import os

urllib3.disable_warnings()
tqdm.pandas()

http_pool = urllib3.PoolManager(500)


def download_image(pair):
    """
    Download an images given its URL and save the normalized version of it
    :param pair: [url to the image, name of the image]
    :return:
    """
    data_dir = "../data/img/train/"
    resize_dim = (224, 224)

    url = pair[0]
    ID = pair[1]

    # check if the image is already there
    save_path = os.path.join(data_dir, str(ID))
    if os.path.isfile(save_path):
        return

    response = http_pool.request("GET", url)
    image = Image.open(io.BytesIO(response.data))
    image = image.convert("RGB")
    image = image.resize(resize_dim)
    image = np.asarray(image, dtype=np.float32)
    image = image / 255

    np.save(save_path, image)


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

    pool = multiprocessing.Pool(processes=50)  # How much can you handle tho?

    with tqdm(total=n_samples, desc="Processing Dataset", unit="images") as pbar:
        for _ in pool.imap_unordered(download_image, pairs):
            pbar.update(1)


if __name__ == "__main__":
    data_dir = "../data/img/train/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download_data("../data/train.json")

