from keras.optimizers import SGD

from networks.loss import *
from utils import params

import numpy as np
from tensorflow.python.lib.io import file_io
import os


callbacks = None

def set_callbacks(new_callbacks):
    global callbacks
    callbacks = new_callbacks


def train_top(generator_train, generator_val, model, base_model,
              steps_per_epoch=None, epochs=5, verbose=1,
              optimizer='rmsprop', validation_steps=None, GCP=False, weights=None, loss=weighted_mean_squared_error, job_dir='./'):
    """
    Trains the top layers of a specified model by freezing ALL base_model layers
    :param generator_train:
    :param generator_val:
    :param model: full model
    :param base_model: base_model
    :param steps_per_epoch
    :param epochs
    :param verbose
    :param optimizer: default rmsprop
    :param validation_steps number of batches of the validation generator used for validation
    :return: history
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    if weights is None:
        weights = np.ones((228,), dtype=int)

    # Freeze all base layers
    for idx_layer, layer in enumerate(model.layers):
        layer.trainable = False if idx_layer < len(base_model.layers) else True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss(weights), metrics=params.metrics)
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks,
                                  max_queue_size=5
                                  )
    if GCP:
        # Save model
        # Save the model locally
        model.save('model.h5')
        file_path = 'model.h5'
        with file_io.FileIO(file_path, mode='rb') as input_f:
            with file_io.FileIO(os.path.join(job_dir, file_path), mode='wb+') as output_f:
                output_f.write(input_f.read())
                print("Saved model.h5 to GCS")

    return history


def train_full(generator_train, generator_val, model,
              steps_per_epoch=None, epochs=5, verbose=1,
              optimizer='rmsprop', validation_steps=None, GCP=False, weights=None, loss=weighted_mean_squared_error, job_dir='./'):
    """
    Train the full model; unfreeze all layers
    :param generator_train:
    :param generator_val:
    :param model:
    :param steps_per_epoch
    :param epochs
    :param verbose
    :param optimizer: default rmsprop
    :param validation_steps number of batches of the validation generator used for validation
    :return: history
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    if weights is None:
        weights = np.ones((228,), dtype=int)

    for layer in model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss(weights), metrics=params.metrics)
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks,
                                  max_queue_size=5
                                  )

    return history


def fine_tune(generator_train, generator_val, model, idx_lower,
              steps_per_epoch=None, epochs=5, verbose=1,
              optimizer=SGD(lr=0.0001, momentum=0.9), validation_steps=None, weights=None,
              loss=weighted_mean_squared_error):
    """
    Fine-tune the model; freeze idx_lower first layers and train
    :param generator_train:
    :param generator_val:
    :param model:
    :param idx_lower: index of the last layer that has to be frozen
    :param steps_per_epoch
    :param epochs
    :param verbose
    :param optimizer: default SGD
    :param validation_steps number of batches of the validation generator used for validation
    :return: history
    """
    if not 0 < idx_lower < len(model.layers):
        raise Exception("idx should be lower than the number of layers")

    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    if weights is None:
        weights = np.ones((228,), dtype=int)

    for layer in model.layers[:idx_lower]:
        layer.trainable = False
    for layer in model.layers[idx_lower:]:
        layer.trainable = True

    # Recompile the model for these modifications to take effect
    model.compile(optimizer=optimizer, loss=loss(weights), metrics=params.metrics)

    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks
                                  )

    return history
