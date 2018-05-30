import numpy as np
from tqdm import tqdm

def loss(models, y_true, weights=None, mean_version=None):
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

    _loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    _loss = np.sum(-np.sum(_loss, axis=0)) / (228 * y_pred.shape[0])
    return _loss

def fitness(y_pred, y_true):
    # Loss
    _loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    _loss = np.sum(-np.sum(_loss, axis=0)) / (228 * y_pred.shape[0])
    return _loss

def particle_optimization(particle, global_opt, local_opt, w, c_1, c_2, v_i):
    '''
    Particle Swarm Optimization
    src: http://www.swarmintelligence.org/tutorials.php
    '''
    r_1 = np.random.rand(1)[0]
    r_2 = np.random.rand(1)[0]
    v_i = (w * v_i) + (c_1 * r_1*(np.array(local_opt) - np.array(particle))) + \
          (c_2*r_2*(np.array(global_opt) - np.array(particle)))
    x_i = np.array(particle) + v_i

    return x_i, v_i

def PSO(predictions, y_true):
    w = .7298
    c_1 = c_2 = 1.49618
    max_iter = 2000
    n_particles = 15

    # Particles equals weights, init local and global opt.
    particles = [np.random.rand(len(predictions)) for x in range(n_particles)]
    # Normalize weights
    particles = [weights / np.sum(weights) for weights in particles]

    # Init global/local fitness, which will be optimized.
    local_opt_fitness = [np.inf for i in range(len(particles))]
    global_opt_fitness = np.inf

    # Init local/global particle optima's (i.e. the weights)
    local_opt = particles
    global_opt = np.random.rand(len(predictions))
    global_opt = global_opt / np.max(global_opt)

    # Parameter which PSO needs.
    v_i = [0 for x in range(n_particles)]

    global_changed_cnt = 0

    # Going to try to optimize the weights such that the loss is minimized.
    for i in tqdm(range(max_iter)):
        for j in range(n_particles):
            # Multiply particles by predictions.
            results = sum([model_prediction * weight for model_prediction, weight in zip(predictions, particles[j])])
            particle_fitness = fitness(results, y_true)

            if particle_fitness < local_opt_fitness[j]:
                local_opt_fitness[j] = particle_fitness
                local_opt[j] = particles[j]
            if particle_fitness < global_opt_fitness:
                global_opt_fitness = particle_fitness
                global_opt = particles[j]
                global_changed_cnt = 0

            # Update weights/particles
            particles[j], v_i[j] = particle_optimization(particles[j], local_opt[j], global_opt, w, c_1, c_2,
                                                         v_i[j])
            # Normalize weights
            particles[j][particles[j] < 0] = 0.0001
            particles[j] = particles[j] / np.sum(particles[j])

        # If global optimum hasn't changed for 1k iterations, then return global opt.
        if global_changed_cnt >= 500:
            return global_opt
        elif global_changed_cnt == 0:
            print('Global optima: {}'.format(global_opt_fitness))
            print('Global optima weights: {}'.format(global_opt))
        else:
            global_changed_cnt += 1


    return global_opt

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
    # list_100k = ['y_pred_incept_higher_100000_model', 'y_pred_incept_lower_100000_model', 'Xception_occ100000', 'y_pred_incept_lower_100000_model']
    # y_pred_avg = np.sum([np.load('./pre-ensemble/val/{}.npy'.format(i)) for i in list_100k], axis=0) / 2
    #
    # # 1. Find weights using validation results.
    # list_models = ['y_pred_incept_all_final_wof1_model_epoch20', 'y_pred_incept_all_final_wof1_model_epoch29', 'y_pred_incept_all_model_best_24_may',
    #                'Xception_full_latest', 'inception_v3_full']
    #
    # y_true = np.load('./pre-ensemble/val/y_true.npy')
    # predictions = [np.load('./pre-ensemble/val/{}.npy'.format(i)) for i in list_models]
    # predictions.append(y_pred_avg)
    # predictions_test = 1
    #
    # # weights = [ 0.01, 0.99,]
    # weights = PSO(predictions, y_true)
    # np.save('./ensemble_results/weights/pso_weights', weights)
    #
    # # 2. Apply weights using test results.
    # #y_pred = ensemble_test_results(predictions, weights)
    # y_pred_v = ensemble_test_results(predictions, weights)
    # print(weights)
    # #np.save('./test/ensemble_results_harm_mean', y_pred_v)
    # np.save('./ensemble_results/val/ensemble_results_PSO.npy', y_pred_v)
    list_100k = ['y_pred_incept_higher_100000_model', 'y_pred_incept_lower_100000_model', 'Xception_occ100000',
                 'y_pred_incept_lower_100000_model']

    # 1. Find weights using validation results.
    list_models = ['y_pred_incept_all_final_wof1_model_epoch20', 'y_pred_incept_all_final_wof1_model_epoch29',
                   'y_pred_incept_all_model_best_24_may', 'Xception_full_latest', 'inception_v3_full']

    y_pred_avg_test = np.sum([np.load('./pre-ensemble/test/{}.npy'.format(i)) for i in list_100k], axis=0) / 2
    predictions_test = [np.load('./pre-ensemble/test/{}.npy'.format(i)) for i in list_models]
    predictions_test.append(y_pred_avg_test)

    weights = np.load('./ensemble_results/weights/pso_weights.npy')
    result = sum([model_prediction * weight for model_prediction, weight in zip(predictions_test, weights)])
    np.save('./ensemble_results/test/ensemble_results_PSO.npy', result)

if __name__ == "__main__":
    main()
