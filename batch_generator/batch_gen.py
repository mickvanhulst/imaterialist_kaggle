import io
import json
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from networks.mobilenet import mobilenet_model

import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http_client = urllib3.PoolManager(50)


class MultiLabelGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(MultiLabelGenerator, self).__init__(*args, **kwargs)

    def make_datagenerator(self, datafile, batch_size=32, dim=(224, 224), n_channels=3, n_classes=228,
                           seed=None, shuffle=True, test=False, data_path='./data/img/', save_images=False):
        return DataGenerator(self, datafile, batch_size, dim, n_channels, n_classes,
                             seed, shuffle, test, data_path, save_images)


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """

    def __init__(self, image_data_generator, datafile, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=228, seed=None, shuffle=True, test=False, data_path='./data/img/', save_images=False):
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

        df['imageId'] = df['imageId'].apply(lambda x: int(x))
        self.df = df

        self.original_indices = df['imageId'].values
        self.epoch_indices = self.original_indices

        # Length Total Dataset
        self.n_samples = len(df)

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

        # load the image from ./data/img/{ID} if it exists
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
        image /= 255

        return image

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
                # image = self.image_data_generator.random_transform(image)
                # image = self.image_data_generator.standardize(image)

                X[i, ] = image

                if not self.test:
                    labels = row['labelId'].values

                    labels = np.asarray(labels)
                    labels = np.subtract(labels[0], 1)

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
