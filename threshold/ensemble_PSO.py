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
    max_iter = 1000
    n_particles = 10

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
            particles[j] = particles[j] / np.sum(particles[j])

        # If global optimum hasn't changed for 1k iterations, then return global opt.
        global_changed_cnt += 1
        if global_changed_cnt >= 1000:
            return global_opt
        elif global_changed_cnt == 0:
            print('Global optima: {}'.format(global_opt_fitness))
            print('Global optima weights: {}'.format(global_opt))


    return global_opt

def main():
    """the best"""
    # 1. Find weights using validation results.
    y_true = np.random.randint(2, size=(228,10000))
    predictions = [np.random.rand(228,10000) for i in range(5)]

    # weights = [ 0.01, 0.99,]
    weights = PSO(predictions, y_true)

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
    #y_pred = ensemble_test_results(predictions, weights)

    # 3. Export predictions and load them using the predictions function in main.py.
    # DON't forget to also use the thresholds there.

if __name__ == "__main__":
    main()
