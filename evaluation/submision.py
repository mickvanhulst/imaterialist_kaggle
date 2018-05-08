import numpy as np


def create_submission(test_generator, model, steps=None, threshold=0.5):
    """
    Create a submission for Kaggle
    :param test_generator:
    :param model:
    :param steps: number of batches to be evaluated, setting this to an other value creates invalid submissions,
                  just here for debugging probably.
    :param threshold: cutoff value for which a label is 1
    """
    preds = model.predict_generator(test_generator,
                                    steps=steps,
                                    verbose=1)

    np.save("preds.npy", preds)

    with open("submission.csv", "w") as f:
        f.write("image_id,label_id\n")
        for i, pred in enumerate(preds):
            positive_samples = np.argwhere(pred > threshold).flatten()
            positive_samples_string = ""
            for positive_sample in positive_samples:
                positive_samples_string = positive_samples_string + " " + str(positive_sample)

            f.write("{},{}\n".format(i, positive_samples_string))
