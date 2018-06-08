import numpy as np
from tqdm import tqdm

from batch_generator.batch_gen_barebones import DataGenerator
from precomputing.networks import xception_model


def create_raw_preds(datafile="../data/train.json", data_path="../data/img_big/train/",
                     save_location="./raw_preds/Xception_train", steps=None, batch_size=32, test=False):
    print("Setting up DataGenerator")
    generator = DataGenerator(datafile=datafile, data_path=data_path, batch_size=batch_size, shuffle=False, test=test)

    base_model = xception_model()

    if steps is None:
        steps = len(generator)

    X_raw = []
    y_raw = []

    for i in tqdm(range(steps), desc="Creating Raw Predictions", unit="batches"):
        if not test:
            X, y = generator[i]
            y_raw.extend(y)
        else:
            X = generator[i]

        X_raw.extend(base_model.predict(X, batch_size=batch_size))

    np.save(save_location + "_X_raw", np.array(X_raw))
    if not test:
        np.save(save_location + "_y_raw", np.array(y_raw))


def create_all():
    """
        Create and save raw predictions for everything
    """
    create_raw_preds(
        datafile="../data/train.json", data_path="../data/img_big/train/",
        save_location="./raw_preds/Xception_train", steps=None
    )
    create_raw_preds(
        datafile="../data/validation.json", data_path="../data/img_big/validation/",
        save_location="./raw_preds/Xception_validation"
    )
    create_raw_preds(
        datafile="../data/test.json", data_path="../data/img_big/test/",
        save_location="./raw_preds/Xception_test", test=True
    )


if __name__ == '__main__':
    create_all()
