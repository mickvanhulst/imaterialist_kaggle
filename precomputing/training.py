import numpy as np
from keras import optimizers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from precomputing.networks import custom_model


def train_classifier(path="./raw_preds/Xception"):
    print("Loading Data...")
    X = np.load(path + "_train_X_raw.npy", mmap_mode='r')
    y = np.load(path + "_train_y_raw.npy")

    X_val = np.load(path + "_validation_X_raw.npy")
    y_val = np.load(path + "_validation_y_raw.npy")

    print("Setting up model...")
    model = custom_model(input_shape=(X.shape[1],),
                         n_outputs=y.shape[1],
                         optimizer=optimizers.Nadam()
                         )

    print("Warmup")
    history = model.fit(X, y,
                        batch_size=128,
                        epochs=1,
                        verbose=2,
                        validation_data=(X_val, y_val),
                        shuffle=True)

    # model = models.load_model("./Xception_top.02-0.087.h5")
    optimizer = optimizers.SGD(lr=1e-2, momentum=0.5, nesterov=True)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        ModelCheckpoint(
            filepath="./custom_conv.{epoch:02d}-{val_loss:.3f}.h5",
            monitor="val_loss",
            verbose=1,
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
                        verbose=2,
                        validation_data=(X_val, y_val),
                        shuffle=True,
                        callbacks=callbacks)


if __name__ == '__main__':
    train_classifier()
