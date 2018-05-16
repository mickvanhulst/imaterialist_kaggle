from keras.models import Model
from keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from networks.training import train_full

from scraper.script_download_image import download_dataset
from keras.layers import Input
from utils import params


def small_convnet(n_outputs, n_features=100, optimizer='rmsprop', input_shape=(200, 200, 3)):
    """
    This method returns a small convnet
    :param n_outputs: number of output classes
    :param n_features: number of hidden units in the feature layer
    :param optimizer: optimizer to be used
    :param input_shape: input shape to the model
    :return: custom inception_v3 model
    """
    input_tensor = Input(shape=input_shape)

    x = Conv2D(128, (2, 2), padding='same', activation='relu')(input_tensor)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    features = Dense(n_features, activation="relu")(x)

    predictions = Dense(n_outputs, activation=params.pred_activation)(features)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=predictions)

    # compile the model
    model.compile(optimizer=optimizer, loss=params.loss)

    model.summary()
    return model, None


if __name__ == "__main__":
    download_dataset('../data/train.json', '../data/img', 1000)

    model, base_model = small_convnet(10)

    # TODO: replace None with generator
    train_full(None, model)
