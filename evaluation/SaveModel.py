from tensorflow.python.lib.io import file_io
import os
from keras.callbacks import Callback


class SaveModel(Callback):
    def __init__(self, job_dir, GCP, vebose=1):
        """
        The F1 score averaged over all classes
        :param validation_generator:
        :param threshold:
        :param steps: number of validation batches, default to whole validation dataset
        """
        super().__init__()
        self.best_F1 = 0
        self.job_dir = job_dir
        self.GCP = GCP
        self.verbose = vebose

    def on_epoch_end(self, epoch, logs={}):
        if logs['F1'] > self.best_F1:
            if self.verbose:
                print("F1 decreased from {} to {}, saving the model...".format(self.best_F1, logs['F1']))
            self.best_F1 = logs['F1']
            if self.GCP:
                # Save model
                # Save the model locally
                self.model.save('model.h5')
                file_path = 'model.h5'
                with file_io.FileIO(file_path, mode='rb') as input_f:
                    with file_io.FileIO(os.path.join(self.job_dir, file_path), mode='wb+') as output_f:
                        output_f.write(input_f.read())
                        print("Saved model.h5 to GCS")
            else:
                self.model.save('best_model.h5')
        elif self.verbose:
            print("F1 did not increase...")
