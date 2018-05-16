from evaluation.f1_score import AveragedF1
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(model_name, test_generator, val_steps=None):
    return [
        ModelCheckpoint('./best_model_{}.h5'.format(model_name),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1),

        ReduceLROnPlateau(monitor='val_loss',
                          patience=1,
                          verbose=1,
                          mode='min'),

        EarlyStopping(monitor='val_loss',
                      min_delta=0.0001,
                      patience=5,
                      mode='min',
                      verbose=1),

        AveragedF1(test_generator, steps=val_steps)
    ]
