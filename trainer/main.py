from batch_generator.batch_gen_barebones import DataGenerator
from networks import training, loss
from networks.inceptionv3 import inception_v3_model
from networks.mobilenet import mobilenet_model
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path

import numpy as np

from networks.xception import xception_model
from threshold.thresholding import thresholding
from utils import params
from utils.load_model import load_model
import argparse
from tensorflow.python.lib.io import file_io

from keras import optimizers


def main(GCP, job_dir):

    if not GCP:
        import matplotlib.pyplot as plt

    print("Setting up Model...")
    global model_name
    global model_class
    global save_images
    model, base_model = model_class(n_classes, input_shape=input_dim)

    if GCP:
        data_folder = 'gs://mlip/'
    else:
        data_folder = '../data/'

    print("Creating Data Generators...")
    training_generator = DataGenerator(
        datafile='{}train.json'.format(data_folder), data_path='{}img/train/'.format(data_folder),
        batch_size=64, shuffle=True)

    validation_generator = DataGenerator(
        datafile='{}validation.json'.format(data_folder), data_path='{}img/validation/'.format(data_folder),
        batch_size=64, shuffle=True)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(
        get_callbacks(
            job_dir,
            None,
            GCP=GCP,
            val_steps=None,
            resume_score=0.12198784052678235,
            # resume_score=0.12167152163024092,  # val_loss of last model
            verbose=1)
    )
    if not GCP:
        # If not training on GCP, try looking for an existing model.
        if os.path.isfile("./model_epoch00.h5".format(model_name)):
            print("Loading existing model ...")
            model = load_model("./model_epoch00.h5".format(model_name))

    #######################################
    #          Top Training               #
    #######################################
    print("Starting Warm-up...")
    # optimizer = optimizers.Nadam()
    #
    # history = training.train_top(generator_train=training_generator, generator_val=validation_generator,
    #                              model=model, base_model=base_model, loss="binary_crossentropy",
    #                              steps_per_epoch=800, epochs=1, optimizer=optimizer, verbose=1)

    print("Starting Top Training...")
    optimizer = optimizers.SGD(lr=1e-2, momentum=1.0, nesterov=True)

    history = training.train_top(generator_train=training_generator, generator_val=validation_generator,
                                 model=model, base_model=base_model, loss="binary_crossentropy",
                                 steps_per_epoch=800, epochs=50, optimizer=optimizer, verbose=1)

    accuracy_train = history.history['acc']
    loss_train = history.history['loss']
    accuracy_val = history.history['val_acc']
    loss_val = history.history['val_loss']
    # categorical_accuracy = history.history['categorical_accuracy']
    # F1 = history.history['F1']
    # thresholds = history.history['threshold']

    best_idx = np.argmin(loss_val)
    print("Best Training Accuracy: {} \t Validation Accuracy: {}".format(np.max(accuracy_train), accuracy_val[best_idx]))
    print("Best Training Loss: {} \t Validation Loss: {}".format(np.max(loss_train), loss_val[best_idx]))
    # # print("Best Categorical Accuracy:", categorical_accuracy[best_idx])
    # print("Best F1:", F1[best_idx])
    # print("Best Thresholds:", thresholds[best_idx])

    #######################################
    #          Fine Tuning                #
    #######################################
    print("Starting Fine Tuning...")
    optimizer = optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)

    history = training.fine_tune(generator_train=training_generator, generator_val=validation_generator,
                                 model=model, idx_lower=115, loss="binary_crossentropy",
                                 steps_per_epoch=160, epochs=100, optimizer=optimizer, verbose=1)

    accuracy_train = history.history['acc']
    loss_train = history.history['loss']
    accuracy_val = history.history['val_acc']
    loss_val = history.history['val_loss']
    # categorical_accuracy = history.history['categorical_accuracy']
    # F1 = history.history['F1']
    # thresholds = history.history['threshold']

    best_idx = np.argmin(loss_val)
    print(
        "Best Training Accuracy: {} \t Validation Accuracy: {}".format(np.max(accuracy_train), accuracy_val[best_idx]))
    print("Best Training Loss: {} \t Validation Loss: {}".format(np.max(loss_train), loss_val[best_idx]))
    # # print("Best Categorical Accuracy:", categorical_accuracy[best_idx])
    # print("Best F1:", F1[best_idx])
    # print("Best Thresholds:", thresholds[best_idx])


def predict():
    global model_name
    global model_class
    global save_images

    # model_name = "./best_model_{}.h5".format(model_name)
    model_name = "../results/models/Xception_full_latest.h5"

    thresholds, _ = thresholding(model_name, model_class,
                                 datafile='../data/validation.json', data_path='../data/img/validation/')

    print("Setting up Test Generator")
    test_generator = DataGenerator(
        datafile='../data/test.json', test=True, shuffle=False, data_path='../data/img/test/')

    print("Setting up Model...")
    if os.path.isfile(model_name):
        print("Loading existing model ...")
        model = load_model(model_name)
    else:
        print("Model '{}' not found!".format(model_name))
        raise Exception("You need to train a model before you can make predictions.")

    create_submission(test_generator, model, steps=None, thresholds=thresholds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--GCP',
        help='Ask user if training on GCP or not',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--job-dir',
        help='Job directory',
        default='./',
        required=False
    )
    args = parser.parse_args()

    if args.GCP and not args.job_dir:
        parser.error("--job-dir should be set if --GCP")

    model_name = "inception_v3_model"
    model_class = inception_v3_model

    save_images = True
    input_dim = (224, 224, 3)
    n_classes = params.n_classes
    label_occ_threshold = 100000
    main(args.GCP, args.job_dir)
    # predict()
