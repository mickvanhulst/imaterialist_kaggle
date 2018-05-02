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


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class MultiLabelGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(MultiLabelGenerator, self).__init__(*args, **kwargs)

    def make_datagenerator(self, datafile, batch_size=32, dim=(224, 224), n_channels=3, n=None,
                             n_classes=228, seed=None, total_batches_seen=0, index_array=None, shuffle=True,
                             ):

        return DataGenerator(datafile, batch_size, dim, n_channels, n,
                             n_classes, seed,total_batches_seen, index_array, shuffle)


class DataGenerator(Iterator):
    'Generates data for Keras'

    def __init__(self, datafile, batch_size=32, dim=(224, 224), n_channels=3, n=None,
                 n_classes=228, seed=None, total_batches_seen=0, index_array=None, shuffle=True,
                 ):
        'Initialization'
        self.n = 0
        self.total_batches_seen = total_batches_seen
        self.seed = seed
        self.index_array = index_array
        self.shuffle = True

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

        # print("len self" , int(np.floor(len(self.train_df) / self.batch_size)))
        return int(np.floor(len(self.train_df) / self.batch_size))

    def next(self):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)

    def download_image(self, url):
        """
        download image from url and reshapes it to dimension size e.g (128x128)
        :returns np array of image dimension
        """
        # print('url' , url)
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url[0])
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb = np.asarray(image_rgb)

        resize_img = cv2.resize(image_rgb, self.dim)
        # print("resize image shape", resize_img.shape)
        # plt.imshow(resize_img)
        # plt.show()

        return resize_img

    # Input list_IDs_temp imageIDs list of size == self.batch_size
    def _get_batches_of_transformed_samples(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            row = self.train_df.loc[self.train_df['imageId'] == int(ID)]
            url = row['url'].values
            labels = row['labelId'].values

            # Store label and class
            X[i,] = self.download_image(url)
            y[i] = labels[0][0]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# training_gen =  DataGenerator(datafile='../data/train.json')
#
# for batch_x, batch_y in training_gen:
#     print(batch_x.shape)
#     print(batch_y.shape)
