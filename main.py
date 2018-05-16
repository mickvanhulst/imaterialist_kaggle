from batch_generator.batch_gen import MultiLabelGenerator
from networks.inceptionv3 import inception_v3_model
from networks.mobilenet import mobilenet_model
from networks import training
from evaluation.callbacks import get_callbacks
from evaluation.submision import create_submission
import os.path
from keras.models import load_model


def train():
    print("Setting up Model...")
    global model_name
    global model_class
    model, base_model = model_class(n_classes, input_shape=input_dim)

    print("Creating Data Generators...")
    training_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    training_generator = training_generator.make_datagenerator(datafile='./data/train.json', save_images=True)

    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=True)
    validation_generator = validation_generator.make_datagenerator(datafile='./data/validation.json', save_images=True)

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(get_callbacks(model_name, validation_generator))
    
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))

    history = training.train_top(generator_train=training_generator, generator_val=validation_generator,
                                 model=model, base_model=base_model,
                                 steps_per_epoch=50, val_percentage=0.05, epochs=10)


def predict():
    print("Setting up Test Generator")
    validation_generator = MultiLabelGenerator(preprocessing_function=model_class, horizontal_flip=False)
    validation_generator = validation_generator.make_datagenerator(datafile='./data/test.json', test=True, shuffle=False)

    print("Setting up Model...")
    global model_name

    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))
    else:
        print("Model 'best_model_{}.h5' not found!".format(model_name))
        raise Exception("You need to train a model before you can make predictions.")

    create_submission(validation_generator, model, steps=1)


if __name__ == "__main__":
    model_name = "mobilenet"
    model_class = mobilenet_model
    input_dim = (224, 224, 3)
    n_classes = 228
    train()
    # predict()
