from evaluation.f1_score import AveragedF1
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(model_name, test_generator, val_steps=None):
    return [
        AveragedF1(test_generator, steps=val_steps),

        ModelCheckpoint('./best_model_{}.h5'.format(model_name),
                        monitor='F1',
                        verbose=1,
                        save_best_only=True,
                        mode='max',
                        period=1),

        ReduceLROnPlateau(monitor='F1',
                          patience=1,
                          verbose=1,
                          mode='max'),

        EarlyStopping(monitor='F1',
                      min_delta=0.0001,
                      patience=5,
                      mode='max',
                      verbose=1)
    ]
