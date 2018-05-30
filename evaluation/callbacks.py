from evaluation.f1_score import AveragedF1
from evaluation.SaveModel import SaveModel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def get_callbacks(job_dir, val_generator, GCP=False, val_steps=None, resume_score=None, verbose=1):

    callbacks = []

    if val_generator is not None:
        callbacks.append(
            AveragedF1(val_generator, steps=val_steps)
        )
        metric = "F1"
        mode = "max"
    else:
        metric = "val_loss"
        mode = "min"

    callbacks.append(
        SaveModel(job_dir, GCP, metric, mode,
                  resume_score=resume_score,
                  verbose=verbose)
    )

    callbacks.append(
        ReduceLROnPlateau(monitor=metric,
                          patience=1,
                          verbose=verbose,
                          mode=mode)
    )

    callbacks.append(
        EarlyStopping(monitor=metric,
                      min_delta=0,
                      patience=10,
                      mode=mode,
                      verbose=verbose)
    )

    return callbacks
