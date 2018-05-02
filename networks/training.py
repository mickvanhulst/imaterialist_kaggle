from keras.optimizers import SGD


def train_top(generator, model, base_model, optimizer='rmsprop'):
    """
    Trains the top layers of a specified model by freezing ALL base_model layers
    :param generator
    :param model: full model
    :param base_model: base_model
    :param optimizer
    :return: history
    """
    # Freeze all base layers
    for idx_layer, layer in enumerate(model.layers):
        layer.trainable = False if idx_layer < len(base_model.layers) else True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # TODO: train the model on the new data for a few epochs
    history = None  # model.fit_generator()

    return history


def train_full(generator, model, optimizer='rmsprop'):
    """
    Train the full model; unfreeze all layers
    :param generator:
    :param model:
    :param optimizer: optimizer to be used
    :return: history
    """
    for layer in model.layers:
        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # TODO: training
    history = None  # model.fit_generator()

    return history


def fine_tune(generator, model, idx_lower, optimizer=SGD(lr=0.0001, momentum=0.9)):
    """
    Fine-tune the model; freeze idx_lower first layers and train
    :param generator:
    :param model:
    :param idx_lower: index of the last layer that has to be frozen
    :param optimizer: default SGD
    :return: history
    """
    if not 0 < idx_lower < len(model.layers):
        raise Exception("idx should be lower than the number of layers")

    for layer in model.layers[:idx_lower]:
        layer.trainable = False
    for layer in model.layers[idx_lower:]:
        layer.trainable = True

    # Recompile the model for these modifications to take effect
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # TODO: train the model again
    history = None  # model.fit_generator()

    return history
