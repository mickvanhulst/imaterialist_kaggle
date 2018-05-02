from keras.callbacks import Callback
from sklearn.metrics import f1_score


class micro_averaged_f1(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_train_begin(self, logs={}):
        self.test = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def on_batch_end(self, batch, logs={}):
        self.test += 5

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        y_pred = self.model.predict(x)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        print(f1_score(self.test_data[1], y_pred, average='micro'))
        