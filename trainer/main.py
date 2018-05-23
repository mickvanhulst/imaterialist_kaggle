from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks import training
from networks.mobilenet import mobilenet_model
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path

#import matplotlib.pyplot as plt
import numpy as np

from utils import params
from utils.load_model import load_model
import argparse
from tensorflow.python.lib.io import file_io

from keras import optimizers

def main(GCP, job_dir):
    print("Setting up Model...")
    global model_name
    global model_class
    global save_images
    model, base_model = model_class(n_classes, input_shape=input_dim)

    if GCP:
        data_folder = 'gs://mlip-test/'
    else:
        data_folder = './data/'

    print("Creating Data Generators...")
    training_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    training_generator = training_generator.make_datagenerator(
        datafile='{}train.json'.format(data_folder), data_path='{}img/train/'.format(data_folder), save_images=save_images,
        label_occ_threshold=label_occ_threshold, batch_size=64, shuffle=True, train=True)

    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    validation_generator = validation_generator.make_datagenerator(
        datafile='{}validation.json'.format(data_folder), data_path='{}img/validation/'.format(data_folder), save_images=save_images,
        batch_size=128, shuffle=True)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(get_callbacks(job_dir, GCP, validation_generator, val_steps=38))

    # if os.path.isfile("./best_model_{}.h5".format(model_name)):
    #     print("Loading existing model ...")
    #     model = load_model("./best_model_{}.h5".format(model_name))

    optimizer = optimizers.Adam()

    #todo: Add job_dir for saving model.
    history = training.train_top(generator_train=training_generator, generator_val=None,
                                 model=model, base_model=base_model,
                                 steps_per_epoch=5, epochs=1, optimizer=optimizer, GCP=GCP, job_dir=job_dir)

    # plt.bar(np.arange(len(training_generator.occurrences)), training_generator.occurrences)
    #
    # plt.title("Class Occurrences During Training")
    # plt.show()



def predict():
    global model_name
    global model_class
    global save_images

    print("Setting up Test Generator")
    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    validation_generator = validation_generator.make_datagenerator(
        datafile='./data/test.json', test=True, shuffle=False, data_path='./data/img/test/', save_images=save_images)

    print("Setting up Model...")
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))
    else:
        print("Model 'best_model_{}.h5' not found!".format(model_name))
        raise Exception("You need to train a model before you can make predictions.")

    create_submission(validation_generator, model, steps=None)

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

    model_name = "mobilenet"
    model_class = mobilenet_model
    save_images = True
    input_dim = (224, 224, 3)
    n_classes = params.n_classes
    label_occ_threshold = 5000
    main(args.GCP, args.job_dir)
    #predict()