import numpy as np
from collections import Counter
import json
import pandas as pd
import os
from PIL import Image
import urllib3
import io

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http_client = urllib3.PoolManager(50)

def get_image(url, ID):
    """
    download image from url, reshapes it to dimension size e.g (128x128) and normalize it
    :returns np array of image dimension
    """
    print('d')
    # load the image from ./data/img/{ID} if it exists
    save_path = os.path.join('../data/img/train/', str(ID) + '.jpg')
    print(save_path)
    if os.path.isfile(save_path):
        image = Image.open(save_path)
        image = np.asarray(image, dtype=K.floatx())
        return image / 255
    print('b')
    response = http_client.request("GET", url[0])
    image = Image.open(io.BytesIO(response.data))
    image = image.convert("RGB")
    #image = image.resize(dim, Image.ANTIALIAS)
    print('c')
    image.save(save_path, optimize=True, quality=85)

    #image = np.asarray(image, dtype=K.floatx())

    return image


test = get_image(['https://contestimg.wish.com/api/webimage/53c7c3c9858d6d0f02ead943-large'], 555)

# Class weights
# https://stackoverflow.com/questions/43481490/keras-class-weights-class-weight-for-one-hot-encoding <-- third post
# https://datascience.stackexchange.com/questions/22170/setting-class-weights-for-categorical-labels-in-keras-using-generator <- voor als we op rij specifiek niveau willen checken.
#https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950 <-- try this666

# https://github.com/keras-team/keras/issues/2115