
import numpy as np
from tqdm import tqdm

# we build a custom classifier to be able to use the 
# sklearn VotingClassifier
def ensemble():
    pass

def loss(models, weights, y_true):
    # todo: get the actual shape of the predictions to fix this function
    # apply the weights to all the models, and sum the predictions
    y_pred = sum([model_prediction * weight for model_prediction, weight in zip(models, weights)])

    # todo maybe we should split the predictions to predictions per sample?
    # currently it is supposed to be all predictions for all validation samples

    _loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    # Should we calculate the mean here for each sample?
    # and then sum those means?

    _loss = np.sum(-np.sum(_loss, axis=0))

    return _loss


def hill_climbing( model_predictions, y_true, iterations=1000):
    """ Try to find optimal weights by looking in the nieghborhood of the
        current solution and only accepting it if it is an improvement
    """
    weights = np.random.rand(len(model_predictions))

    # normaliseer naar een probability distribution
    weights /= np.sum(weights)
    current_loss = loss(model_predictions, weights, y_true)

    for i in tqdm(range(iterations)):
        new_weights = weights + np.random.normal(scale=0.1, size=len(weights))
        new_weights[new_weights<0] = 0.0001
        new_weights /= np.sum(new_weights)



        new_loss = loss(model_predictions, new_weights, y_true)

        if new_loss < current_loss:
            current_loss = new_loss
            weights = new_weights

    return weights

def main():
    """the best"""
    y_true = np.random.rand(228,10000)
    predictions = [np.random.rand(228,10000) for i in range(5)]


    # weights = [ 0.01, 0.99,]
    print(hill_climbing(predictions, y_true))


if __name__ == "__main__":
    main()
