from keras.optimizers import SGD

callbacks = None


def set_callbacks(new_callbacks):
    global callbacks
    callbacks = new_callbacks


def train_top(generator_train, generator_val, model, base_model,
              steps_per_epoch=None, epochs=5, verbose=1,
              optimizer='rmsprop', val_percentage=0.5):
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
    :param val_percentage percentage of the validation set used per epoch
    :return: history
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    # Freeze all base layers
    for idx_layer, layer in enumerate(model.layers):
        layer.trainable = False if idx_layer < len(base_model.layers) else True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=int(len(generator_val)*val_percentage),
                                  callbacks=callbacks
                                  )
    return history


def train_full(generator_train, generator_val, model,
               steps_per_epoch=None, epochs=5, verbose=1,
               optimizer='rmsprop', val_percentage=0.5):
    """
    Train the full model; unfreeze all layers
    :param generator_train:
    :param generator_val:
    :param model:
    :param steps_per_epoch
    :param epochs
    :param verbose
    :param optimizer: default rmsprop
    :param val_percentage percentage of the validation set used per epoch
    :return: history
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    for layer in model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=int(len(generator_val)*val_percentage),
                                  callbacks=callbacks
                                  )
    return history


def fine_tune(generator_train, generator_val, model, idx_lower,
              steps_per_epoch=None, epochs=5, verbose=1,
              optimizer=SGD(lr=0.0001, momentum=0.9), val_percentage=0.5):
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
    :param val_percentage percentage of the validation set used per epoch
    :return: history
    """
    if not 0 < idx_lower < len(model.layers):
        raise Exception("idx should be lower than the number of layers")

    if steps_per_epoch is None:
        steps_per_epoch = len(generator_train)

    for layer in model.layers[:idx_lower]:
        layer.trainable = False
    for layer in model.layers[idx_lower:]:
        layer.trainable = True

    # Recompile the model for these modifications to take effect
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_val,
                                  validation_steps=int(len(generator_val)*val_percentage),
                                  callbacks=callbacks
                                  )
    return history
