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

        # Rewrite y_pred/y_true cause @Dennis was too tired :P.
        y_pred = np.reshape(y_pred, [self.val_gen.batch_size * self.steps, 228])
        y_true = np.reshape(y_true, [self.val_gen.batch_size * self.steps, 228])

        # Calculate threshold per class
        thresholds = np.linspace(0.0, 1.0, num=50)
        best_thresh_list = []
        print('started calculating F1')

        for n_class in range(228):
            y_true_temp = y_true[:, n_class]
            y_pred_temp = y_pred[:, n_class]

            high_score = 0
            for threshold in thresholds:
                y_pred_thresh = y_pred_temp.copy()

                y_pred_thresh[y_pred_thresh >= threshold] = 1
                y_pred_thresh[y_pred_thresh < threshold] = 0

                # Compare prediction with true labels.
                score = (y_true_temp == y_pred_thresh).sum()

                if score > high_score:
                    best_threshold = threshold
                    high_score = score
                # print("high score ", high_score, "threshold ", best_threshold)
            best_thresh_list.append(best_threshold)

        # Now apply thresholds to y_predict
        for i, val in enumerate(y_pred):
            y_pred[i][y_pred[i] >= best_thresh_list] = 1
            y_pred[i][y_pred[i] < best_thresh_list] = 0

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        F1 = f1_score(y_true.flatten(), y_pred.flatten())
        self.f1_scores.append(F1)

        logs['F1'] = F1
        print('F1-score: {}'.format(logs['F1']))

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
