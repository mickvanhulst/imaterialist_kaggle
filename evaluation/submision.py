import numpy as np


def create_submission(test_generator, model, thresholds=None, steps=None):
    """
    Create a submission for Kaggle using a single model
    :param test_generator:
    :param model:
    :param steps: number of batches to be evaluated, setting this to an other value creates invalid submissions,
                  just here for debugging probably.
    :param thresholds: cutoff value for which a label is 1
    """
    if thresholds is None:
        thresholds = [0.5 for _ in range(228)]

    preds = model.predict_generator(test_generator,
                                    steps=steps,
                                    verbose=1)

    np.save("preds.npy", preds)
    #preds = np.load(test_generator)
    with open("submission.csv", "w") as f:
        f.write("image_id,label_id\n")
        for i, pred in enumerate(preds):
            positive_samples = np.argwhere(pred > thresholds).flatten()
            # In keras, they are 0-based, but we need them 1 based
            positive_samples += 1

            positive_samples_string = ""
            for positive_sample in positive_samples:
                positive_samples_string = positive_samples_string + " " + str(positive_sample)

            f.write("{}, {}\n".format(i+1, positive_samples_string))

#thresholds = np.load('../threshold/ensemble_results/thresholds/mean.npy')
#create_submission('../threshold/ensemble_results/test/mean.npy', thresholds=thresholds)