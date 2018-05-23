from keras.callbacks import Callback
from sklearn.metrics import f1_score
#from IPython.display import clear_output
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.pylab import ion
from tqdm import tqdm


class AveragedF1(Callback):
    def __init__(self, validation_generator, threshold=0.5, steps=None):
        """
        The F1 score averaged over all classes
        :param validation_generator:
        :param threshold:
        :param steps: number of validation batches, default to whole validation dataset
        """
        super().__init__()
        self.val_gen = validation_generator
        self.threshold = threshold
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
            for step in tqdm(range(self.steps), desc="F1-Score", disable=not self.params['verbose']):
                n_x, n_y = self.val_gen[step]
                y_true[step * self.val_gen.batch_size * self.val_gen.n_classes:
                       (step + 1) * self.val_gen.batch_size * self.val_gen.n_classes] \
                    = n_y.flatten()
                y_pred[step * self.val_gen.batch_size * self.val_gen.n_classes:
                       (step + 1) * self.val_gen.batch_size * self.val_gen.n_classes] \
                    = self.model.predict(n_x).flatten()

        y_pred[y_pred > self.threshold] = 1
        y_pred[y_pred <= self.threshold] = 0
        score = f1_score(y_true, y_pred, average='macro')
        self.f1_scores.append(score)
        logs['F1'] = score
        print('F1-score: {}'.format(score))

    def on_train_end(self, logs={}):
        self.plot()

    def plot(self):
        print('Had to disable this for GCP, can save image later.')
        # clear_output()
        # plt.plot(self.f1_scores)
        # plt.ylim([0, 1])
        # plt.title("F1 Score")
        # plt.ylabel("F1")
        # plt.xlabel("epoch")
        # plt.show()
