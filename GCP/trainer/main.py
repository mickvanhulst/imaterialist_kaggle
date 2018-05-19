from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.inceptionv3 import inception_v3_model
from networks.mobilenet import mobilenet_model
from networks.resnet_50 import resnet_50_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path

import matplotlib.pyplot as plt
import numpy as np

from utils import params
from utils.load_model import load_model
import argparse
from tensorflow.python.lib.io import file_io

from keras import optimizers

def train(train_files='gs://mlip/train.json', job_dir='gs://mlip-test/mlip-test15',
          validation_files='gs://mlip/validation.json', **args):
    print("Setting up Model...")
    global model_name
    global model_class
    global save_images
    model, base_model = model_class(n_classes, input_shape=input_dim)

    print("Creating Data Generators...")
    training_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    training_generator = training_generator.make_datagenerator(
        datafile=train_files, data_path=job_dir + 'img/train/', save_images=save_images,
        label_occ_threshold=label_occ_threshold, batch_size=64, shuffle=True, train=True)

    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    validation_generator = validation_generator.make_datagenerator(
        datafile=validation_files, data_path=job_dir + 'img/validation/', save_images=save_images,
        batch_size=128, shuffle=True)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(get_callbacks(model_name, validation_generator, val_steps=38))

    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))

    optimizer = optimizers.Adam()

    history = training.train_top(generator_train=training_generator, generator_val=None, job_dir=job_dir,
                                 model=model, base_model=base_model,
                                 steps_per_epoch=50, epochs=10, optimizer=optimizer, GCP=False)


    plt.bar(np.arange(len(training_generator.occurrences)), training_generator.occurrences)

    plt.title("Class Occurrences During Training")
    plt.show()


def predict():
    global model_name
    global model_class
    global save_images

    print("Setting up Test Generator")
    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    validation_generator = validation_generator.make_datagenerator(
        datafile='../../data/test.json', test=True, shuffle=False, data_path='../../data/img/test/', save_images=save_images)

    print("Setting up Model...")
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))
    else:
        print("Model 'best_model_{}.h5' not found!".format(model_name))
        raise Exception("You need to train a model before you can make predictions.")

    create_submission(validation_generator, model, steps=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--validation-files',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    print('arguments are: '.format(arguments))
    save_images = True
    input_dim = (224, 224, 3)
    n_classes = 229
    label_occ_threshold = 5000

    train(**arguments)
