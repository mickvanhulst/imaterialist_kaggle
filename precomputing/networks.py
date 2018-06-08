from keras import Model
from keras.applications import Xception
import keras.layers as L


def xception_model(optimizer='SGD', input_shape=(299, 299, 3)):
    """
    This method returns the Xception model without top layers
    """
    # create the base pre-trained model
    base_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False)

    # Extract raw features
    x = base_model.output
    raw_features = L.GlobalAveragePooling2D(name="raw_features")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=raw_features)

    # freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # compile the model such that it may be used
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    model.summary()
    return model


def custom_model(input_shape=(2048,), n_outputs=228, optimizer="SGD"):
    input = L.Input(shape=input_shape)

    x = L.Dense(1024, activation='relu')(input)
    x = L.BatchNormalization()(x)

    x = L.Dense(512, activation='relu')(x)
    x = L.BatchNormalization()(x)

    x = L.Dense(256, activation='relu')(x)
    x = L.BatchNormalization()(x)

    preds = L.Dense(n_outputs, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=preds)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    return model


if __name__ == "__main__":
    xception_model()
    custom_model()
