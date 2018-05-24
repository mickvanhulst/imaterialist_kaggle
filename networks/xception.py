from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from utils import params


def xception_model(n_outputs, n_features=1024, optimizer='rmsprop', input_shape=(299, 299, 3)):
    """
    This method returns the inception_v3 model with custom TOP layers
    All layers except the TOP are frozen (i.e. cannot be trained)
    :param n_outputs: number of output classes
    :param n_features: number of hidden units in the feature layer
    :param optimizer: optimizer to be used
    :param input_shape: input shape to the model
    :return: custom inception_v3 model
    """
    # create the base pre-trained model
    base_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D(name="features_pooling")(x)
    x = Dense(n_features, activation='relu', name="features")(x)

    # Prediction layer, Probability per class
    predictions = Dense(n_outputs, activation=params.pred_activation, name="predictions")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all Xception layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=params.loss)

    model.summary()
    return model, base_model


if __name__ == "__main__":
    xception_model(params.n_classes, input_shape=params.input_shape)
