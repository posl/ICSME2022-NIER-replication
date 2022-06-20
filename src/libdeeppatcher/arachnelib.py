import sys, os
sys.path.append(os.path.abspath('..'))
import keras
import numpy as np
import random
from functools import reduce
from typing import List
from libdeeppatcher.types import Dataset, Weight
import time


def printfunc(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print('[{}] Start ===>'.format(func.__name__))
        result = func(*args, **kwargs)
        print('\n[{}]<=== End '.format(func.__name__))
        print('time: {:.2}'.format(time.time() - start_time))
        return result
    return wrapper


def f_chain(chian_initializer, *funcs):
    return reduce(lambda x,f: f(x), funcs, chian_initializer)


def _generate_input_data(dataset, batch_size):
    data_size = len(dataset.data)
    steps_per_epoch = int((data_size-1) / batch_size) + 1
    while True:
        for batch_num in range(steps_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, data_size)
            x = dataset.data[start:end]
            y = dataset.label[start:end]
            yield Dataset(x,y)


def identify_pareto(scores: np.ndarray) -> List[int]:
    """Identify pareto front indices.
    cf. https://pythonhealthcare.org/tag/pareto-front/
    :param scores:
        Each item has two scores
        eg: np.array([[1,2],
                      [3,4],
                      [5,6]])
    :return: pareto front indices
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item.
    # This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i'
                # (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front].tolist()


def _get_velocity_bounds(model):
    """Get velocity bounds.

    "W is the set of all neural weights
    between our target layer and the preceding one."

    wb = np.max(all_weights) - np.min(all_weights)
    vb = (wb / 5, wb * 5)

    :param model:
    :return: dictionary whose key is layer index
             and value is velocity bounds
    """
    # Range from 1: Use preceding layer
    # To #layers-1: Except for final output layer
    velocity_bounds = {}
#     for layer_index in range(1, len(model.layers) - 1):
    for layer_index in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_index)

        # Target only trainable layers
        if not layer.trainable:
            continue
        # Out of scope if layer does not have kernel
        if not hasattr(layer, 'kernel'):
            continue

        # Get all weights
        all_weights = []
        target_weights = layer.get_weights()[0]
        for j in range(target_weights.shape[1]):
            for i in range(target_weights.shape[0]):
                all_weights.append(target_weights[i][j])

        pre_layer = model.get_layer(index=layer_index-1)
        # Flatten layer has no kernel weights
        if hasattr(pre_layer.get_weights, 'kernel'):
            pre_target_weights = pre_layer.get_weights()[0]
            for j in range(pre_target_weights.shape[1]):
                for i in range(pre_target_weights.shape[0]):
                    all_weights.append(pre_target_weights[i][j])

        # Velocity bounds defined at equations 5 and 6
        wb = np.max(all_weights) - np.min(all_weights)
        vb = (wb / 5, wb * 5)

        velocity_bounds[layer_index] = vb

    return velocity_bounds


def _update_position(x, v):
    return x + v


def _update_velocity(x, v, p, g, vb, layer_index):
    """Update velocity.
    :param x: current position
    :param v: current velocity
    :param p: personal best position
    :param g: global best position
    :param vb: velocity bounds computed in each layer
    :param layer_index:
    :return: new velocity
    """
    # "We follow the general recommendation
    # in the literature and set both to 4.1"
    phi = 4.1
    # "Equation 3"
    chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
    # "Equation 2"
    ro1 = random.uniform(0, phi)
    ro2 = random.uniform(0, phi)
    # TODO Using same value 'chi'
    #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
    new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))
    "we additionally set velocity bounds"
    for n in range(len(new_v)):
        _vb = vb[layer_index[n]]
        _new_v = np.abs(new_v[n])
        _sign = 1 if 0 < new_v[n] else -1
        if _new_v < _vb[0]:
            new_v[n] = _vb[0] * _sign
        if _vb[1] < _new_v:
            new_v[n] = _vb[1] * _sign
    return new_v


def _count_correct_pred(model, dataset: Dataset):
    """Count the number of dataset predicted correctly.
    """
    actual = model.predict(dataset.data, verbose=0)
    expected = dataset.label
    return sum(actual.argmax(axis=1) == expected.argmax(axis=1))


def _calculate_loss(model, dataset: Dataset, batch_size=32):
    dsize = len(dataset.label)
    generator = _generate_input_data(dataset, batch_size)
    steps_per_epoch = int((dsize-1)/batch_size) + 1
    history = model.fit_generator(generator=generator,
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=0)
    loss = history.history['loss'][0]
    return loss


def load_and_mutate(model, weights):
    mutated_model = _mutate(model, weights)
    return mutated_model


def _mutate(model, locations: List[Weight]):
    """Mutate model with location.
    """
    model_clone = keras.models.clone_model(model)
    model_clone.compile(optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=model.metrics)
    model_clone.set_weights(model.get_weights())
    
    for location in locations:
        (layer_index, nw_i, nw_j), val = location
        layer = model_clone.get_layer(index=layer_index)
        weights = layer.get_weights()
        weights[0][nw_i][nw_j] = val
        layer.set_weights(weights)
        
    return model_clone


if __name__ == '__main__':
    scores = np.array([
        [97, 23],
        [55, 77],
        [34, 76],
        [80, 60],
        [99,  4],
        [81,  5],
        [ 5, 81],
        [30, 79],
        [15, 80],
        [70, 65],
        [90, 40],
        [40, 30],
        [30, 40],
        [20, 60],
        [60, 50],
        [20, 20],
        [30,  1],
        [60, 40],
        [70, 25],
        [44, 62],
        [55, 55],
        [55, 10],
        [15, 45],
        [83, 22],
        [76, 46],
        [56, 32],
        [45, 55],
        [10, 70],
        [10, 30],
        [79, 50],
    ])
    
    actual_pareto = identify_pareto(scores)
    expected_pareto = [0,1,3,4,6,7,8,9,10]
    assert actual_pareto == expected_pareto

    actual_pareto_front = scores[actual_pareto]
    expected_pareto_front = np.array([
        [97, 23],
        [55, 77],
        [80, 60],
        [99,  4],
        [5 , 81],
        [30, 79],
        [15, 80],
        [70, 65],
        [90, 40],
    ])
    assert actual_pareto_front.all() == expected_pareto_front.all()
