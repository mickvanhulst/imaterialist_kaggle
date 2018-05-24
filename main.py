from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.mobilenet import mobilenet_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path

import networks.loss as loss

import matplotlib.pyplot as plt
import numpy as np

from utils import params
from utils.load_model import load_model

from keras import optimizers


def train():
    print("Setting up Model...")
    global model_name
    global model_class
    global save_images

    model, base_model = model_class(n_classes, input_shape=input_dim)

    print("Creating Data Generators...")
    training_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    training_generator = training_generator.make_datagenerator(
        datafile='./data/train.json', data_path='./data/img/train/', save_images=save_images,
        label_occ_threshold=label_occ_threshold, batch_size=32, shuffle=True, train=True)

    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    validation_generator = validation_generator.make_datagenerator(
        datafile='./data/validation.json', data_path='./data/img/validation/', save_images=save_images,
        batch_size=32, shuffle=True)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    # About half of the validation set will be used per epoch
    training.set_callbacks(get_callbacks(model_name, validation_generator, val_steps=10))
    
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))

    """
        LR:
        Default: 1e-3
        Last: 1e-5
    """
    optimizer = optimizers.Adam(lr=1e-3)

    # weights = np.loadtxt("class_weights.csv", delimiter=',')
    weights = np.ones((228,))
    weights[65] = 0.25

    # history = training.train_top(generator_train=training_generator, generator_val=None,
    #                              model=model, base_model=base_model, loss=loss.weighted_categorical_crossentropy,
    #                              steps_per_epoch=1, epochs=10, optimizer=optimizer)

    # history = training.fine_tune(generator_train=training_generator, generator_val=None,
    #                              model=model, idx_lower=55, loss=loss.weighted_categorical_crossentropy,
    #                              steps_per_epoch=2, epochs=10, optimizer=optimizer)

    history = training.train_full(generator_train=training_generator, generator_val=None,
                                  model=model, weights=weights, loss=loss.weighted_categorical_crossentropy,
                                  steps_per_epoch=10, epochs=2, optimizer=optimizer)

    plt.bar(np.arange(len(training_generator.occurrences)), training_generator.occurrences)

    plt.title("Class Occurrences During Training")
    plt.show()

    print(history)
    F1 = history['F1']
    thresholds = history['threshold']
    print("Best F1:", F1[np.argmax(F1)])
    print("Best Thresholds:", thresholds[np.argmax(F1)])



def predict():
    global model_name
    global model_class
    global save_images

    print("Setting up Test Generator")
    test_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    test_generator = test_generator.make_datagenerator(
        datafile='./data/test.json', data_path='./data/img/test/', save_images=save_images,
        test=True, shuffle=False, batch_size=128)

    print("Setting up Model...")
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))
    else:
        print("Model 'best_model_{}.h5' not found!".format(model_name))
        raise Exception("You need to train a model before you can make predictions.")

    create_submission(test_generator, model, steps=None)


if __name__ == "__main__":
    model_name = "mobilenet"
    model_class = mobilenet_model
    save_images = False
    input_dim = (224, 224, 3)
    n_classes = params.n_classes
    label_occ_threshold = 5000
    train()
    # predict()
