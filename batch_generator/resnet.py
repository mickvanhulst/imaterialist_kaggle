import keras.layers
from keras.applications.imagenet_utils import preprocess_input
from batch_generator import batch_gen
from keras import layers
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Flatten, Dense, Dropout
from evaluation import f1_score
import numpy as np
from sklearn import metrics




def resnet_50():
    resnet = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3),
                                         pooling=None, classes=228)

    for layer in resnet.layers:
        if (layer.name == 'res5c_branch2a'):
            break
        layer.trainable = False

    x = resnet.output
    x = Flatten()(x)
    x = Dense(500, activation='relu', name='d300')(x)
    x = Dropout(0.2)(x)
    output = Dense(228, activation='sigmoid', name='predict_layer')(x)





    return Model(input=[resnet.input], output=[output])



def resnet_prep(x):
    original_shape = x.shape
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.squeeze(x, axis=0)


def create_callbacks(val_gen):
    checkpoint = ModelCheckpoint('./best_model1.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=5,
                               mode='min',
                               verbose=1)

    callback_list = [
    checkpoint, early_stop, f1_score.AveragedF1(val_gen)
    ]
    return callback_list


def main():


    model = resnet_50()
    # model.summary()

    training_generator_dummy = batch_gen.MultiLabelGenerator(preprocessing_function=resnet_50,
                                                             horizontal_flip=True)
    validation_generator_dummy = batch_gen.MultiLabelGenerator(preprocessing_function=resnet_50)

    training_generator = training_generator_dummy.make_datagenerator(datafile='../data/train.json')
    validation_generator = validation_generator_dummy.make_datagenerator(datafile='../data/validation.json')


    # model.load_weights('./best_model1.h5')
    model.compile(optimizer=SGD(lr=0.02), loss='binary_crossentropy', metrics=['accuracy'])
    calls = create_callbacks(validation_generator)
    # history = model.fit_generator(generator = training_generator,
    #                               steps_per_epoch  = 1,
    #                               epochs           = 5,
    #                               verbose          = 1,
    #                               validation_data  = validation_generator,
    #                               validation_steps = 2,
    #                               callbacks        = calls
    #                               )

    # y_predict = model.predict_generator(validation_generator, steps = 10)
    #

    # for batch_x, batch_y in(training_generator):
    #     y_predict = model.predict(batch_x)
    #     y_predict = np.subtract(1,y_predict)
    #
    #     log_loss = metrics.log_loss(batch_y, y_predict)
    #     print(" log loss : " ,log_loss)



    #
    # y_predict = model.predict(batch_x)
    #
    # log_loss = metrics.log_loss(batch_y, y_predict)
    # print(log_loss)
    # # #
    # #
    # # # print(y_predict)
    # # # y_predict = y_predict[y_predict > 0.2]
    # # #
    # for i in range(2):
    #     print((y_predict[i]))
    #     print(batch_y[i])
    #     print("geile opmaak ----")

if __name__ == "__main__":
    main()
