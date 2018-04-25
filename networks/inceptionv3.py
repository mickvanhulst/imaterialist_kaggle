from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.optimizers import SGD

from scraper.script_download_image import download_dataset


def inception_plus(n_outputs):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Feature Layer
    x = Dense(1024, activation='relu')(x)
    # Prediction layer, Probability per class
    predictions = Dense(n_outputs, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.summary()
    return model, base_model


def train_model(model, base_model):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # TODO: train the model on the new data for a few epochs
    # model.fit_generator()

    # second: fine-tuning convolutional layers from inception V3. Freeze
    # the bottom N layers and train the remaining top layers.

    # Visualize layer names and indices to see how many layers we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # TODO: train the model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers
    # model.fit_generator()

    return model


if __name__ == "__main__":
    download_dataset('../data/train.json', '../data/img', 1000)

    model, base_model = inception_plus(100)
    train_model(model, base_model)
