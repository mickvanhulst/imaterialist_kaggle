import numpy as np
from keras import optimizers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from precomputing.networks import custom_model


def predict(path="./raw_preds/Xception_test", model_path="./custom_conv.15-0.087.h5"):
    print("Loading Data...")

    X = np.load(path + "_X_raw.npy", mmap_mode='r')  # mmap streams data instead of loading into ram

    print("Loading Model")
    model = models.load_model(model_path)

    preds = model.predict(X,
                          batch_size=128,
                          verbose=1)

    print("Saving Preds")
    postfix = "test" if "test" in path else "validation"
    np.save("./final_preds/Xception_{}.npy".format(postfix), preds)

    print("Done!")


if __name__ == '__main__':
    predict(path="./raw_preds/Xception_test")
    predict(path="./raw_preds/Xception_validation")
