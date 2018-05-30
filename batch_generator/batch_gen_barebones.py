import json
import os

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import urllib3
from PIL import Image
from tqdm import tqdm

from utils import params

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http_client = urllib3.PoolManager(500)


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """

    def __init__(self, datafile, batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=params.n_classes, shuffle=True, test=False, data_path='./data/img/'):
        """ Initialization """
        self.test = test
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes

        # vars for saving and loading the image
        self.path = data_path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(datafile, 'r') as f:
            train_data = json.load(f)

        df = pd.DataFrame.from_records(train_data["images"], index="imageId")

        if not test:
            labels_df = pd.DataFrame.from_records(train_data["annotations"], index="imageId")
            df = pd.concat([df, labels_df], axis=1)
            df["labelId"] = df["labelId"].apply(lambda row: self._labels_to_array([int(i) for i in row]))

        # Shape of train_df ['imageId', 'url', 'labelId'], shape: (1014544, 3)
        self.df = df

        self.indices = self._get_existing_indices()

        # Length Total Dataset
        self.n_samples = len(self.indices)

        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Create new indices for the epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _get_existing_indices(self):
        """
        Check the data path which images are there
        These may differ from the indices in the dataframe
        :return:
        """
        if self.test:  # All test images are there
            return self.df.index.tolist()

        generator_name = "train_idxs" if "train" in self.path else "val_idxs"
        idx_path = os.path.join(self.path, generator_name + ".npy")
        if os.path.isfile(idx_path):
            return np.load(idx_path)

        indices = []
        for idx in tqdm(self.df.index.tolist(), desc="Validating Dataset", unit="images"):
            img_path = os.path.join(self.path, str(idx) + '.jpg')
            if os.path.isfile(img_path):
                indices.append(idx)

        np.save(idx_path, indices)
        return indices

    def _labels_to_array(self, labels):
        """
        Converts a list of labels to a one-hot encoded array
        """
        labels_array = np.zeros((self.n_classes,), dtype=int)
        labels = np.array(labels)

        # Labels are 1-based, so do - 1
        labels_array[labels - 1] = 1

        return labels_array

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self._get_data(batch_indices)

    def get_image(self, ID):
        """
        download image from url, reshapes it to dimension size e.g (128x128) and normalize it
        :returns np array of image dimension
        """
        # load the image from ./data/img/set/{ID} if it exists
        img_path = os.path.join(self.path, str(ID) + '.jpg')

        if os.path.isfile(img_path):
            image = Image.open(img_path)
            image = np.asarray(image, dtype=K.floatx())
            return image / 255
        else:
            raise Exception("Label should not be here!")

    def _get_data(self, idxs):
        """
        Generates data containing batch_size samples
        :param idxs
        :return X: (n_samples, *dim, n_channels) y:(n_samples, n_classes)
        """
        # Initialization
        X = np.empty((len(idxs), *self.dim, self.n_channels), dtype=K.floatx())

        if not self.test:
            y = np.empty((len(idxs), self.n_classes), dtype=np.int)

        # Generate data
        for i, idx in enumerate(idxs):
            try:
                row = self.df.loc[idx]

                image = self.get_image(idx)
                X[i, ] = image

                if not self.test:
                    y[i, ] = row['labelId']
            except Exception as e:
                print("Exception|", e, "|", row['url'])

        if not self.test:
            return X, y
        else:
            return X


def test():
    generator = DataGenerator(datafile='../data/validation.json', data_path='../data/img/validation/', shuffle=True)

    n_samples = 0
    for i in tqdm(range(len(generator)), desc="Iterating Over Generator", unit="batches"):
        batch_x, _ = generator[i]
        n_samples += batch_x.shape[0]

    print("Total Samples:", n_samples)
    print("Expected Samples:", generator.n_samples)
    assert n_samples == generator.n_samples


if __name__ == "__main__":
    test()

    # model, _ = mobilenet_model(generator.n_classes)
    # model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1)
