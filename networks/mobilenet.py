from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from networks.training import train_top, fine_tune

from scraper.script_download_image import download_dataset
from keras.layers import Input


def mobilenet_model(n_outputs, n_features=1024, optimizer='rmsprop', input_shape=(299, 299, 3)):
    """
    This method returns the mobilenet model with custom TOP layers
    All layers except the TOP are frozen (i.e. cannot be trained)
    :param n_outputs: number of output classes
    :param n_features: number of hidden units in the feature layer
    :param optimizer: optimizer to be used
    :param input_shape: input shape to the model
    :return: custom mobilenet model
    """
    # create the base pre-trained model
    base_model = MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Feature Layer
    x = Dense(n_features, activation='relu')(x)
    # Prediction layer, Probability per class
    predictions = Dense(n_outputs, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all MobileNet layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    model.summary()
    return model, base_model


def test():
    download_dataset('../data/train.json', '../data/img', 1000)

    model, base_model = mobilenet_model(100)

    # TODO: replace None with generator
    train_top(None, model, base_model)
    # train the top 2 inception blocks, i.e. we will freeze the first 249 layers
    fine_tune(None, model, idx_lower=249)


if __name__ == "__main__":
    test()
