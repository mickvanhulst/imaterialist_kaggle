from evaluation.f1_score import AveragedF1
from evaluation.SaveModel import SaveModel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def get_callbacks(job_dir, val_generator, GCP=False, val_steps=None, verbose=1):

    callbacks = []

    if val_generator is not None:
        callbacks.append(
            AveragedF1(val_generator, steps=val_steps)
        )
        metric = "F1"
    else:
        metric = "categorical_accuracy"

    callbacks.append(
        SaveModel(job_dir, GCP)
    )

    callbacks.append(
        ReduceLROnPlateau(monitor=metric,
                          patience=1,
                          verbose=verbose,
                          mode='max')
    )

    callbacks.append(
        EarlyStopping(monitor=metric,
                      min_delta=0.0001,
                      patience=5,
                      mode='max',
                      verbose=verbose)
    )

    return callbacks
