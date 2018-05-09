from keras.callbacks import Callback
from sklearn.metrics import f1_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import ion


class MicroAveragedF1(Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.val_gen = validation_generator
        self.f1_score = []
        self.batches = 2
        self.test = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def on_train_begin(self, logs={}):
        self.test = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def on_epoch_end(self, epoch, logs={}):
        y_true = np.zeros(32*self.batches*228)
        y_pred = np.zeros(32*self.batches*228)
        for i in range(self.batches):
            n_x, n_y = self.val_gen.next()
            y_true[i*32*228:(i+1)*32*228] = n_y.flatten()
            y_pred[i*32*228:(i+1)*32*228] = self.model.predict(n_x).flatten()
        print(y_pred.shape)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        score = f1_score(y_true, y_pred, average='micro')
        self.f1_score.append(score)
        print('F1-score: {}'.format(score))

    def on_train_end(self, logs={}):
        self.plot()

    def plot(self):
        clear_output()
        N = len(self.f1_score)
        plt.plot(range(0, N), self.f1_score)
        plt.legend('micro F1-score')
        plt.show()
