from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_recall_curve
#from IPython.display import clear_output
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.pylab import ion
from tqdm import tqdm


class AveragedF1(Callback):
    def __init__(self, validation_generator, steps=None):
        """
        The F1 score averaged over all classes
        :param validation_generator:
        :param threshold:
        :param steps: number of validation batches, default to whole validation dataset
        """
        super().__init__()
        self.val_gen = validation_generator
        self.f1_scores = []
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):
        if not self.steps:
            y_true = self.val_gen.df['labelId'].values
            y_true = np.array([self.val_gen._labels_to_array(row) for row in y_true])
            y_pred = self.model.predict_generator(self.val_gen, verbose=self.params['verbose'], max_queue_size=5)
        else:
            y_true = np.zeros(self.val_gen.batch_size * self.steps * self.val_gen.n_classes)
            y_pred = np.zeros(self.val_gen.batch_size * self.steps * self.val_gen.n_classes)
            for step in tqdm(range(self.steps),
                             desc="F1-Score", disable=False if self.params['verbose'] == 1 else True, unit="batches"):
                n_x, n_y = self.val_gen[step]
                y_true[step * self.val_gen.batch_size * self.val_gen.n_classes:
                       (step + 1) * self.val_gen.batch_size * self.val_gen.n_classes] \
                    = n_y.flatten()
                y_pred[step * self.val_gen.batch_size * self.val_gen.n_classes:
                       (step + 1) * self.val_gen.batch_size * self.val_gen.n_classes] \
                    = self.model.predict(n_x).flatten()

        y_true = np.array(y_true, dtype=int)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        F1 = 2 * (precision * recall) / (precision + recall)
        F1 = [f if not np.isnan(f) else 0 for f in F1]
        best_idx = np.argmax(F1)
        F1 = F1[best_idx]
        self.f1_scores.append(F1)

        logs['threshold'] = thresholds[best_idx]
        logs['F1'] = F1
        print('F1-score: {}\nThreshold: {}'.format(logs['F1'], logs['threshold']))

    def on_train_end(self, logs={}):
        self.plot()

    def plot(self):
        pass
        #todo: 'Had to disable this for GCP, can save graphs as imags
        # clear_output()
        # plt.plot(self.f1_scores)
        # plt.ylim([0, 1])
        # plt.title("F1 Score")
        # plt.ylabel("F1")
        # plt.xlabel("epoch")
        # plt.show()
