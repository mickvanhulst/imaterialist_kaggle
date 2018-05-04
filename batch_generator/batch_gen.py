import numpy as np
import keras
import json
import pandas as pd
from PIL import Image
from urllib3.util import Retry
import urllib3
import io
import cv2
from keras.preprocessing.image import ImageDataGenerator, Iterator
import matplotlib.pyplot as plt
import keras.backend as K
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class MultiLabelGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(MultiLabelGenerator, self).__init__(*args, **kwargs)

    def make_datagenerator(self, datafile, batch_size=32, dim=(224, 224), n_channels=3, n=None,
                           n_classes=228, seed=None, total_batches_seen=0, index_array=None, shuffle=True):

        return DataGenerator(self, datafile, batch_size, dim, n_channels, n,
                             n_classes, seed,total_batches_seen, index_array, shuffle)


class DataGenerator(Iterator):
    'Generates data for Keras'

    def __init__(self,image_data_generator, datafile, batch_size=32, dim=(224, 224), n_channels=3, n=None,
                 n_classes=228, seed=None, total_batches_seen=0, index_array=None, shuffle=True):
        'Initialization'
        self.n = 0
        self.total_batches_seen = total_batches_seen
        self.seed = seed
        self.index_array = index_array
        self.shuffle = True
        self.image_data_generator = image_data_generator


        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.shuffle = shuffle
        # self.on_epoch_end()
        with open(datafile, 'r') as f:
            train_data = json.load(f)

        train_imgs_df = pd.DataFrame.from_records(train_data["images"])
        train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
        train_labels_df["labelId"] = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
        train_df = pd.merge(train_imgs_df, train_labels_df, on="imageId", how="outer")
        train_df['imageId'] = train_df['imageId'].apply(lambda x: int(x))

        # Shape of train_df ['imageId', 'url', 'labelId'], shape: (1014544, 3)
        self.train_df = train_df

        # Length Total Dataset
        self.samples = len(train_df)

        super(DataGenerator, self).__init__(self.samples, batch_size, shuffle, seed)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_df) / self.batch_size))

    def next(self):
        'Generate one batch of data'
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)

    def download_image(self, url):
        """
        download image from url and reshapes it to dimension size e.g (128x128)
        :returns np array of image dimension
        """
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url[0])
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        resize_img = image_rgb.resize(self.dim)
        image_numpy = np.asarray(resize_img, dtype=K.floatx())


        return image_numpy

    def _labels_to_array(self, labels):
        labels_array = np.zeros(self.n_classes)
        labels_array[labels] = 1
        return labels_array

    # Input list_IDs_temp imageIDs list of size == self.batch_size
    def _get_batches_of_transformed_samples(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=K.floatx())
        y = np.empty((self.batch_size,self.n_classes), dtype = np.int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            row = self.train_df.loc[self.train_df['imageId'] == int(ID)]
            url = row['url'].values
            labels = row['labelId'].values


            labels = np.asarray(labels)
            labels = np.subtract(labels[0],1)


            image = self.download_image(url)
            image = self.image_data_generator.random_transform(image)
            image = self.image_data_generator.standardize(image)



            X[i,] = image

            # Store label and class
            y[i,] = self._labels_to_array(labels)



        # plt.show()
        return X, y

# training_gen =  MultiLabelGenerator(datafile='../data/train.json')
#
# training_generator_dummy = MultiLabelGenerator(horizontal_flip=True)
#
# training_generator = training_generator_dummy.make_datagenerator(datafile='../data/train.json')
#
# # for batch_x, batch_y in training_generator:
#     print(batch_x.shape)
#     print(batch_y.shape)
