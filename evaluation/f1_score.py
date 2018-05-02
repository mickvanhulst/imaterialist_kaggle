from keras.callbacks import Callback
from sklearn.metrics import f1_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.pylab import ion


class micro_averaged_f1(Callback):
    def __init__(self, validation_generator):
        self.val_gen = validation_generator
        self.f1_score = []

    def on_train_begin(self, logs={}):
        self.test = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0


    def on_epoch_end(self, epoch, logs={}):
        x = []
        y = []
        for i in range(0, 5):
            n_x, n_y = self.val_gen.next()
            x.extend(n_x)
            y.extend(n_y)
        y_pred = self.model.predict(x)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        score = f1_score(self.test_data[1], y_pred, average='micro')
        self.f1_score.append(score)
        print(score)

    def on_train_end(self, logs={}):
        self.plot()

    def plot(self):
        clear_output()
        N = len(self.f1_score)
        plt.plot(range(0, N), self.f1_score)
        plt.legend('micro F1-score')
        plt.show()
