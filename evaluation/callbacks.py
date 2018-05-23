from evaluation.f1_score import AveragedF1
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(model_name, test_generator, val_steps=None):

    callbacks = []

    if test_generator is not None:
        callbacks.append(
            AveragedF1(test_generator, steps=val_steps))
        metric = "F1"
    else:
        metric = "categorical_accuracy"

    callbacks.append(
        # todo: make variable from path using GCP + job-dir
        ModelCheckpoint('gs://mlip-test/test12332/best_model_{}.h5'.format(model_name),
                        monitor=metric,
                        verbose=1,
                        save_best_only=True,
                        mode='max',
                        period=1)
    )

    callbacks.append(
        ReduceLROnPlateau(monitor=metric,
                          patience=1,
                          verbose=1,
                          mode='max')
    )

    callbacks.append(
        EarlyStopping(monitor=metric,
                      min_delta=0.0001,
                      patience=5,
                      mode='max',
                      verbose=1)
    )

    return callbacks
