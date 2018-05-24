import io
import json
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.python.lib.io import file_io
from tqdm import tqdm

from networks.mobilenet import mobilenet_model

import os

from utils import params

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http_client = urllib3.PoolManager(500)


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """

    def __init__(self, datafile, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=params.n_classes, shuffle=True, test=False, data_path='./data/img/',
                 save_images=True, GCP=True):
        """ Initialization """
        self.n = 0
        self.test = test
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes

        # vars for saving and loading the image
        self.save_images = save_images
        self.path = data_path

        self.occurrences = np.zeros((n_classes,), dtype=int)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if GCP:
            with file_io.FileIO(datafile, 'r') as f:
                train_data = json.load(f)
        else:
            with open(datafile, 'r') as f:
                train_data = json.load(f)

        df = pd.DataFrame.from_records(train_data["images"])

        if not test:
            train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
            df = pd.merge(df, train_labels_df, on="imageId", how="outer")
            df["labelId"] = df["labelId"].apply(lambda x: [int(i) for i in x])

        df['imageId'] = df['imageId'].astype(int, copy=False)

        # Shape of train_df ['imageId', 'url', 'labelId'], shape: (1014544, 3)
        self.df = df

        self.original_indices = self.df['imageId'].values
        self.epoch_indices = self.original_indices

        # Length Total Dataset
        self.n_samples = len(self.df)

        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Create new indices for the epoch
        """
        if self.shuffle:
            np.random.shuffle(self.epoch_indices)

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        batch_indices = self.epoch_indices[index * self.batch_size:(index + 1) * self.batch_size]

        return self._get_batches_of_transformed_samples(batch_indices)

    def get_image(self, url, ID):
        """
        download image from url, reshapes it to dimension size e.g (128x128) and normalize it
        :returns np array of image dimension
        """
        # load the image from ./data/img/set/{ID} if it exists
        save_path = os.path.join(self.path, str(ID) + '.jpg')

        if os.path.isfile(save_path):
            image = Image.open(save_path)
            image = np.asarray(image, dtype=K.floatx())
            return image / 255

        response = http_client.request("GET", url[0])
        image = Image.open(io.BytesIO(response.data))
        image = image.convert("RGB")
        image = image.resize(self.dim, Image.ANTIALIAS)

        if self.save_images:
            image.save(save_path, optimize=True, quality=85)

        image = np.asarray(image, dtype=K.floatx())

        return image / 255

    def _labels_to_array(self, labels):
        """
        Converts a list of labels to a one-hot encoded array
        """
        labels_array = np.zeros((self.n_classes,), dtype=int)
        labels = np.array(labels)

        # Labels are 1-based, so do - 1
        if len(labels) > 0:
            labels_array[labels - 1] = 1

        # Bookkeeping
        self.occurrences += labels_array

        return labels_array

    # Input list_IDs_temp imageIDs list of size == self.batch_size
    def _get_batches_of_transformed_samples(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        :param list_IDs_temp
        :return X: (n_samples, *dim, n_channels) y:(n_samples, n_classes)
        """
        # Initialization
        X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels), dtype=K.floatx())

        if not self.test:
            y = np.empty((len(list_IDs_temp), self.n_classes), dtype=np.int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            row = self.df.loc[self.df['imageId'] == int(ID)]
            url = row['url'].values
            image = self.get_image(url, ID)
            X[i, ] = image

            if not self.test:
                # The 0 indexing is because row is a mini dataframe, so it is two dimensional.
                labels = row['labelId'].values[0]
                labels = np.asarray(labels)
                # Store label and class
                y[i, ] = self._labels_to_array(labels)
        if not self.test:
            return X, y
        else:
            return X


if __name__ == "__main__":
    generator = DataGenerator(
        datafile='../data/validation.json', data_path='../data/img/validation/', save_images=True, shuffle=True)

    n_samples = 0
    for i in tqdm(range(len(generator)), desc="Iterating Over Generator", unit="batches"):
        batch_x, _ = generator[i]
        n_samples += batch_x.shape[0]

    print("Total Samples:", n_samples)

    # model, _ = mobilenet_model(generator.n_classes)
    # model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1)
