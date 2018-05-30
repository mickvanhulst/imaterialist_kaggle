import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from precomputing.networks import custom_model


def train_classifier(path="./raw_preds/Xception"):
    print("Loading Data...")
    X = np.load(path + "_train_X_raw.npy")
    y = np.load(path + "_train_y_raw.npy")

    X_val = np.load(path + "_validation_X_raw.npy")
    y_val = np.load(path + "_validation_y_raw.npy")

    print("Setting up model...")
    model = custom_model(input_shape=(X.shape[1],),
                         n_outputs=y.shape[1],
                         optimizer=optimizers.SGD(lr=1e-2, momentum=1.0, nesterov=True))

    callbacks = [
        ModelCheckpoint(
            filepath="./Xception_top.{epoch:02d}-{val_loss:.3f}.h5",
            monitor="val_loss",
            verbose=2,
            save_best_only=True,
            mode="min"
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min"
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            verbose=1,
            factor=0.2,
            patience=3,
            mode="min"
        )
    ]

    history = model.fit(X, y,
                        batch_size=128,
                        epochs=100,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        shuffle=True,
                        callbacks=callbacks)


if __name__ == '__main__':
    train_classifier()
