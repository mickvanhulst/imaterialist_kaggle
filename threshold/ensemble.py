
import numpy as np
from tqdm import tqdm
from batch_generator.batch_gen_weights import MultiLabelGenerator

def loss(models, y_true, weights=None, mean_version=None):
    # todo: get the actual shape of the predictions to fix this function
    # apply the weights to all the results, and sum the predictions
    if weights is not None:
        y_pred = sum([model_prediction * weight for model_prediction, weight in zip(models, weights)])
    else:
        if mean_version == 'harm_mean':
            y_pred = len(models) / np.sum([1.0 / model for model in models], axis=0)
        elif mean_version == 'mean':
            y_pred = np.mean(models, axis=0)
        else:
            # In this case the output is just one model.
            y_pred = models

    # todo maybe we should split the predictions to predictions per sample?
    # currently it is supposed to be all predictions for all validation samples
    _loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    # Should we calculate the mean here for each sample?
    # and then sum those means?

    _loss = np.sum(-np.sum(_loss, axis=0)) / (228 * y_pred.shape[0])
    return _loss

def hill_climbing(model_predictions, y_true, iterations=1000):
    """ Try to find optimal weights by looking in the neighborhood of the
        current solution and only accepting it if it is an improvement
    """
    weights = np.random.rand(len(model_predictions))

    # normaliseer naar een probability distribution
    weights /= np.sum(weights)
    current_loss = loss(model_predictions, y_true, weights)

    for i in tqdm(range(iterations)):
        new_weights = weights + np.random.normal(scale=0.01, size=len(weights))
        new_weights[new_weights<0] = 0.0001
        new_weights /= np.sum(new_weights)

        new_loss = loss(model_predictions, y_true, new_weights)

        if new_loss < current_loss:
            current_loss = new_loss
            weights = new_weights

    return weights

def ensemble_test_results(model_results, weights=None, mean_version='harm_mean'):
    # Use weights and test data set results to ensemble.
    if weights is not None:
        # Weighted avg
        return sum([model_prediction * weight for model_prediction, weight in zip(model_results, weights)])
    else:
        if mean_version == 'harm_mean':
            # Calculate harmonic mean (only defined for positive real numbers).
            return len(model_results) / np.sum([1.0 / res for res in model_results], axis=0)

        else:
            # Mean
            return np.mean(model_results, axis=0)

def main():
    """the best"""
    # 1. Find weights using validation results.
    y_true = np.random.randint(2, size=(228,10000))
    predictions = [np.random.rand(228,10000) for i in range(5)]
    predictions_test = 1

    # weights = [ 0.01, 0.99,]
    weights = hill_climbing(predictions, y_true)

    ensemble_loss = loss(predictions, y_true, weights)
    ensemble_loss_harm = loss(predictions, y_true, mean_version='harm_mean')
    ensemble_loss_mean = loss(predictions, y_true, mean_version='mean')
    pred_loss = [loss(i, y_true) for i in predictions]

    print('Ensemble loss weighted mean: {}'.format(ensemble_loss))
    print('Ensemble loss harmonic mean: {}'.format(ensemble_loss_harm))
    print('Ensemble loss normal mean: {}'.format(ensemble_loss_mean))

    for i in range(len(pred_loss)):
       print('Loss prediction {}: {}'.format(i, pred_loss[i]))
    print(weights)
    # 2. Apply weights using test results.
    y_pred = ensemble_test_results(predictions, weights)

    # 3. Export predictions and load them using the predictions function in main.py.
    # DON't forget to also use the thresholds there.
    harmonic_mean_res = ensemble_test_results(predictions_test)
    mean_res = ensemble_test_results(predictions_test, mean_version='mean')
    weighted_mean = ensemble_test_results(predictions_test, weights)

    np.save('./ensemble_results_harm_mean', harmonic_mean_res)
    np.save('./ensemble_results_mean', mean_res)
    np.save('./ensemble_results_hc', weighted_mean)

if __name__ == "__main__":
    main()
