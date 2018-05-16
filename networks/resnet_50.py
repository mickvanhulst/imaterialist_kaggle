from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from networks.training import train_top, fine_tune

from scraper.script_download_image import download_dataset
from utils import params


def resnet_50_model(n_outputs, n_features=1024, optimizer='rmsprop', input_shape=(299, 299, 3)):
    """
    This method returns the resnet_50 model with custom TOP layers
    All layers except the TOP are frozen (i.e. cannot be trained)
    :param n_outputs: number of output classes
    :param n_features: number of hidden units in the feature layer
    :param optimizer: optimizer to be used
    :param input_shape: input shape to the model
    :return: custom inception_v3 model
    """

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None, classes=n_outputs)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_features, activation='relu', name='features')(x)
    preds = Dense(228, activation=params.pred_activation, name='predict_layer')(x)

    model = Model(inputs=[base_model.input], outputs=[preds])

    # Freeze layers
    for layer in model.layers:
        if layer.name == 'res5c_branch2a':
            break
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=params.loss)

    model.summary()
    return model, base_model


def test():
    download_dataset('../data/train.json', '../data/img', 1000)

    model, base_model = resnet_50_model(100)

    # TODO: replace None with generator
    train_top(None, model, base_model)
    # Freeze up to res5c_branch2a
    fine_tune(None, model, idx_lower=163)


if __name__ == "__main__":
    test()
