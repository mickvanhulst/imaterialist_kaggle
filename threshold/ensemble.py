
import numpy as np
from tqdm import tqdm

# def hill_climbing(model_predictions, y_true, iterations=1000):
#     """ Try to find optimal weights by looking in the neighborhood of the
#         current solution and only accepting it if it is an improvement
#     """
#     weights = np.random.rand(len(model_predictions))
#
#     # normaliseer naar een probability distribution
#     weights /= np.sum(weights)
#     current_loss = loss(model_predictions, y_true, weights)
#
#     for i in tqdm(range(iterations)):
#         new_weights = weights + np.random.normal(scale=0.01, size=len(weights))
#         new_weights[new_weights<0] = 0.0001
#         new_weights /= np.sum(new_weights)
#
#         new_loss = loss(model_predictions, y_true, new_weights)
#
#         if new_loss < current_loss:
#             current_loss = new_loss
#             weights = new_weights
#
#     return weights

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
    # 0. Average >100.000 <100.000
    list_100k = ['y_pred_incept_higher_100000_model', 'y_pred_incept_lower_100000_model', 'Xception_occ100000',
                 'y_pred_incept_lower_100000_model']
    y_pred_avg = np.sum([np.load('./pre-ensemble/val/{}.npy'.format(i)) for i in list_100k], axis=0) / 2

    # 1. Find weights using validation results.
    list_models = ['y_pred_incept_all_final_wof1_model_epoch20', 'y_pred_incept_all_final_wof1_model_epoch29',
                   'y_pred_incept_all_model_best_24_may', 'Xception_full_latest', 'inception_v3_full']

    predictions = [np.load('./pre-ensemble/val/{}.npy'.format(i)) for i in list_models]
    y_true = np.load('./pre-ensemble/val/y_true.npy')
    predictions.append(y_pred_avg)

    y_pred_avg_test = np.sum([np.load('./pre-ensemble/test/{}.npy'.format(i)) for i in list_100k], axis=0) / 2
    predictions_test = [np.load('./pre-ensemble/test/{}.npy'.format(i)) for i in list_models]
    predictions_test.append(y_pred_avg_test)

    ensemble_loss_harm = loss(predictions, y_true, mean_version='harm_mean')
    ensemble_loss_mean = loss(predictions, y_true, mean_version='mean')
    pred_loss = [loss(i, y_true) for i in predictions]

    print('Ensemble loss harmonic mean: {}'.format(ensemble_loss_harm))
    print('Ensemble loss normal mean: {}'.format(ensemble_loss_mean))

    for i in range(len(pred_loss)):
       print('Loss prediction {}: {}'.format(i, pred_loss[i]))

    # 3. Export predictions and load them using the predictions function in main.py.
    harmonic_mean_res = ensemble_test_results(predictions_test)
    mean_res = ensemble_test_results(predictions_test, mean_version='mean')

    harmonic_mean_res_v = ensemble_test_results(predictions)
    mean_res_v = ensemble_test_results(predictions, mean_version='mean')

    np.save('./ensemble_results/test/harm_mean.npy', harmonic_mean_res)
    np.save('./ensemble_results/test/mean.npy', mean_res)

    np.save('./ensemble_results/val/harm_mean.npy', harmonic_mean_res_v)
    np.save('./ensemble_results/val/mean.npy', mean_res_v)
    #np.save('./val/y_true', y_true)

if __name__ == "__main__":
    main()
