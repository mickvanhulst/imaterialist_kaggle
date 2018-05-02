from batch_generator.batch_gen import DataGenerator
from networks.inceptionv3 import inception_v3_model
from networks import training
from evaluation.callbacks import get_callbacks
import os.path
from keras.models import load_model


def main():
    print("Setting up Model...")
    model_name = "inception_v3"
    model, base_model = inception_v3_model(n_classes, input_shape=input_dim)

    print("Creating Data Generators...")
    training_generator = DataGenerator(datafile='./data/train.json')
    validation_generator = DataGenerator(datafile='./data/validation.json')

    print("Training batches:", len(training_generator))
    print("Validation batches:", len(validation_generator))

    training.set_callbacks(get_callbacks("model_name", None))
    
    if os.path.isfile("./best_model_{}.h5".format(model_name)):
        print("Loading existing model ...")
        model = load_model("./best_model_{}.h5".format(model_name))

    history = training.train_top(generator_train=training_generator, generator_val=validation_generator,
                                 model=model, base_model=base_model,
                                 steps_per_epoch=50, val_percentage=0.05)


if __name__ == "__main__":
    input_dim = (224, 224, 3)
    n_classes = 228
    main()
