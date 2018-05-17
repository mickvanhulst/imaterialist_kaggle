import io
import json
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, Iterator
from tqdm import tqdm
from urllib3.util import Retry
from collections import Counter


from networks.mobilenet import mobilenet_model

import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http_client = urllib3.PoolManager(50)


class MultiLabelGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(MultiLabelGenerator, self).__init__(*args, **kwargs)

    def make_datagenerator(self, datafile, batch_size=32, dim=(224, 224), n_channels=3, n_classes=229,
                           seed=None, shuffle=True, test=False, data_path='./data/img/', save_images=False, train=False, label_occ_threshold=5000):
        return DataGenerator(self, datafile, batch_size, dim, n_channels, n_classes,
                             seed, shuffle, test, train, data_path, save_images, label_occ_threshold)


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """

    def __init__(self, image_data_generator, datafile, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=229, seed=None, shuffle=True, test=False, train=False, data_path='./data/img/', save_images=False, label_occ_threshold=5000):
        """ Initialization """
        self.n = 0
        self.test = test
        self.seed = seed
        self.shuffle = shuffle
        self.image_data_generator = image_data_generator

        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes

        # vars for saving and loading the image
        self.save_images = save_images
        self.path = data_path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(datafile, 'r') as f:
            train_data = json.load(f)

        df = pd.DataFrame.from_records(train_data["images"])

        if not test:
            train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
            df = pd.merge(df, train_labels_df, on="imageId", how="outer")
            df["labelId"] = df["labelId"].apply(lambda x: [int(i) for i in x])

        df['imageId'] = df['imageId'].astype(int)

        # Remove infrequent classes for training dataset and save class weights.
        if train:
            df["labelId"] = self._find_freq_classes(df["labelId"], label_occ_threshold)
            self.class_weights = self._get_class_weights(df["labelId"])
            factor = 1.0 / sum(self.class_weights.values())
            self.class_weights_normalized = {k: v * factor for k, v in self.class_weights.items()}

        # Shape of train_df ['imageId', 'url', 'labelId'], shape: (1014544, 3)
        self.df = df

        self.original_indices = self.df['imageId'].values
        self.epoch_indices = self.original_indices

        # Length Total Dataset
        self.n_samples = len(self.df)

        self.on_epoch_end()

    def _find_freq_classes(self, series, label_occ_threshold):
        # Get labels to be ignored.
        total_list = []
        for i, item in series.iteritems():
            total_list.extend(item)
        count_items = Counter(total_list)
        labels_whitelist = [x for x in count_items if count_items[x] > label_occ_threshold]

        return series.apply(lambda x: [i for i in x if i in labels_whitelist])

    def _get_class_weights(self, series):
        # Create binary encoding for the labels
        y = np.empty((len(series), self.n_classes), dtype=np.int)

        for i, item in series.iteritems():
            labels = np.asarray(item)

            # Store label and class
            y[i,] = self._labels_to_array(labels)

        # Count per column
        counter = y.sum(axis=0)

        # Calculate and return weights
        majority = np.max(counter)
        class_weights = {i: 0 if counter[i] == 0 else float(majority / counter[i]) for i in range(len(counter))}

        return class_weights

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

    # def next(self):
    #     """ Generate one batch of data """
    #     with self.lock:
    #         index_array = next(self.index_generator)
    #
    #     return self._get_batches_of_transformed_samples(index_array)

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
        labels_array = np.zeros(self.n_classes)
        labels_array[labels] = 1
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
            try:
                row = self.df.loc[self.df['imageId'] == int(ID)]
                url = row['url'].values
                image = self.get_image(url, ID)
                X[i, ] = image

                if not self.test:
                    # TODO: for some weird reason this is a list of lists, probably bug somewhere, but meh :P.
                    labels = row['labelId'].values[0]
                    labels = np.asarray(labels)
                    # Store label and class
                    y[i, ] = self._labels_to_array(labels)
            except Exception as e:
                 print("Exception|", e, "|", url)

        if not self.test:
            return X, y
        else:
            return X

if __name__ == "__main__":
    generator = MultiLabelGenerator(horizontal_flip=True)
    generator = generator.make_datagenerator(
        datafile='../data/validation.json', data_path='../data/img/validation/', save_images=True, shuffle=True)

    # n_samples = 0
    # for i in tqdm(range(len(generator)), desc="Iterating Over Generator", unit="batches"):
    #     batch_x, _ = generator[i]
    #     n_samples += batch_x.shape[0]
    #
    # print("Total Samples:", n_samples)

    model, _ = mobilenet_model(generator.n_classes)
    model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1)
