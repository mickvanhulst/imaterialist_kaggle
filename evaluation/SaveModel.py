from tensorflow.python.lib.io import file_io
import os
from keras.callbacks import Callback
import operator


class SaveModel(Callback):
    def __init__(self, job_dir, GCP, metric, mode, vebose=1):
        """
        The F1 score averaged over all classes
        :param validation_generator:
        :param threshold:
        :param steps: number of validation batches, default to whole validation dataset
        """
        super().__init__()
        self.best_score = float('inf') if mode is "min" else -float('inf')
        self.job_dir = job_dir
        self.GCP = GCP
        self.verbose = vebose
        self.metric = metric
        self.comparison = operator.lt if mode is "min" else operator.gt

    def on_epoch_end(self, epoch, logs={}):
        if self.comparison(logs[self.metric], self.best_score):
            if self.verbose:
                print("{} decreased from {} to {}, saving the model...".format(
                    self.metric, self.best_score, logs[self.metric]))
            self.best_score = logs[self.metric]

            # Save model
            # Save the model locally
            file_path = 'model_epoch{:02d}.h5'.format(epoch)
            self.model.save(file_path)
            if self.GCP:
                # Save the model to GCS
                with file_io.FileIO(file_path, mode='rb') as input_f:
                    with file_io.FileIO(os.path.join(self.job_dir, file_path), mode='wb+') as output_f:
                        output_f.write(input_f.read())
                        print("Saved {} to GCS".format(file_path))
        elif self.verbose:
            print("{} did not increase...".format(self.metric))
