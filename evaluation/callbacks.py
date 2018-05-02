from evaluation.f1_score import MicroAveragedF1
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(model_name, test_data):
    return [
        ModelCheckpoint('./best_model_{}.h5'.format(model_name),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1),

        EarlyStopping(monitor='val_loss',
                      min_delta=0.0001,
                      patience=5,
                      mode='min',
                      verbose=1),

        MicroAveragedF1(test_data)
    ]
