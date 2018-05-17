from batch_generator.batch_gen_weights import MultiLabelGenerator
from networks.inceptionv3 import inception_v3_model
from networks.mobilenet import mobilenet_model
from networks.resnet_50 import resnet_50_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path
from utils.load_model import load_model
from collections import Counter
import json
import pandas as pd
import numpy as np

def train():
    print("Setting up Model...")
    global model_name
    global model_class
    global save_images
    model, base_model = model_class(n_classes, input_shape=input_dim)

    print("Creating Data Generators...")
    training_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    training_generator = training_generator.make_datagenerator(
        datafile='./data/train.json', data_path='./data/img/train/', save_images=save_images, label_occ_threshold=label_occ_threshold, train=True)

    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    validation_generator = validation_generator.make_datagenerator(
        datafile='./data/validation.json', data_path='./data/img/validation/', save_images=save_images)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(get_callbacks(model_name, validation_generator, val_steps=3))
    
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))

    history = training.train_top(generator_train=training_generator, generator_val=validation_generator,
                                 model=model, base_model=base_model,
                                 steps_per_epoch=None, epochs=10)

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

    create_submission(validation_generator, model, steps=1)


if __name__ == "__main__":
    model_name = "resnet_50"
    model_class = mobilenet_model
    save_images = True
    input_dim = (224, 224, 3)
    n_classes = 229
    label_occ_threshold = 5000
    train()
    # predict()
