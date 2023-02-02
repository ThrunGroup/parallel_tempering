import random
import math
import kmedoids
import numpy as np
import argparse
import copy
import sklearn.metrics


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print overall loss comparison (default: True)",
    )

def total_dist(points: list, medoids: np.array,) -> int:
    """
  Adds up the closest_dist for each point with its closest medoid

  :param points: list of all the points (including the medoids)
  :param medoids: numpy array of the medoids
  :return: the sum of smallest Euclidean distances for all points
  """
    min_dists = [np.min(np.linalg.norm(medoids - i, axis=1), axis=0) for i in points]
    # import ipdb; ipdb.set_trace()
    return np.sum(min_dists)


def remove_array(list_arr: np.array, arr: np.array) -> None:
    """
    Removes the array arr from the list of arrays (list_arr)
    :param list_arr: a list consisting of several numpy arrays
    :param arr: the array to be removed
    :return: None
    """
    ind = 0
    size = len(list_arr)
    while ind != size and not np.array_equal(list_arr[ind], arr):
        ind += 1
    if ind != size:
        list_arr.pop(ind)
    else:
        raise ValueError("array not found in list.")


def swap_medoids(medoids: np.array, points: list, T: float,) -> np.array:
    """
  Returns the best state and its respective loss for which the total Euclidean distance is smallest

  :param medoids: array of the medoids
  :param T: represents the temperature, affecting the model's transition probability
  :param points: list of all the points (including the medoids)
  :return: the set of medoids after the swap with largest transition probability
  """
    non_medoids = []
    for i in points:
        if not np.any(np.all(i == medoids, axis=1)):
            non_medoids.append(i)

    m = random.choice(medoids)
    n_m = random.choice(non_medoids)

    # new_state represents the set of medoids
    new_state = copy.deepcopy(medoids)
    new_state.append(n_m)
    remove_array(new_state, m)
    # f(x') or f_new_state is just total_dist(points, state)
    initial_f = total_dist(points, medoids)
    f_new_state = total_dist(points, new_state)
    # make the swap with a probability proportional to the transition probability
    tp = min(1, np.float128((math.e) ** (((-1 / T) * (initial_f - f_new_state)))))
    if random.uniform(0, 1) < tp:
        medoids = copy.deepcopy(new_state)

    total_loss = f_new_state
    return medoids, total_loss


def find_medoids(
    points: list, T: float, possible_medoids: dict, swap_count: int,
) -> np.array:
    """
    Returns the set of medoids after swapping for a particular value of T

    :param points: list of all the points (including the medoids)
    :param T: represents the temperature, affecting the model's transition probability
    :param possible_medoids: a dictionary that keeps track of the set of medoids for each temperature T
    :param swap_count: the number of swaps that have already been performed
    :return: the array of medoids after the swap
    """

    # run chains for various values of T  (temperature)
    # T is geometrically defined (T, 2T, 4T, 8T, etc.)
    # medoids_p is the set of medoids prior to swapping
    # for the initialization, medoids is the set prior to swapping
    medoids_p = possible_medoids[T]

    # medoids_a is the set of medoids after swapping
    medoids_a, pt_total_loss = swap_medoids(medoids_p, points, T)
    swap_count += 1
    return medoids_a, pt_total_loss, swap_count


def swap_temp(possible_medoids: dict, loss: dict, T_values: list) -> int:
    """
    Completes all possible swaps sets corresponding to a higher temperature and
    a lower temperature, where the higher temperature has a lower loss than the
    lower temperature

    :param possible_medoids: a dictionary that keeps track of the set of medoids for each temperature T
    :param loss: a dictionary with the total losses respective to each temperature
    :param T_values: a list of all temperatures for each running chain
    """

    # if the loss with higher temperature is greater than the loss with lower temperature, we swap the set of medoids and the loss
    for temp in T_values:
        temp_idx = T_values.index(temp)
        for greater_temp in T_values[temp_idx + 1 :]:
            if loss[temp] < loss[greater_temp]:
                possible_medoids[temp], possible_medoids[greater_temp] = (
                    possible_medoids[greater_temp],
                    possible_medoids[temp],
                )
                loss[temp], loss[greater_temp] = loss[greater_temp], loss[temp]
    lowest_loss = min(loss.values())
    return (lowest_loss, possible_medoids, loss)


def main(points: list, T: float, k: int, conv_condition: int, num_temp : int) -> int:
    """
  Returns k medoids for a set of points depending on a certain temperature

  :param points: list of all the points (including the medoids)
  :param T: represents the temperature, affecting the model's transition probability
  :param k: the number of medoids in a sample of points
  :param num_temp: the number of temperature values
  :return: the best state for medoids and its corresponding loss
  """
    args = parse_args()

    # parameter keeps track of the number of swaps
    swap_count = 0

    # medoids currently represents the initialization of choosing the medoids
    medoids = random.sample(points, k)

    # dictionaries containing the set of medoids and its respective overall loss for each value of T
    possible_medoids = {}
    loss = {}
    T_values = [T * (2 ** i) for i in range(num_temp)]
    # initially, the medoids for each temperature chain is just the random sample generated above
    for i in T_values:
        possible_medoids[i] = medoids

    same = 0
    medoids_p = medoids
    while same < conv_condition:
        for temp in T_values:
            possible_medoids[temp], loss[temp], swap_count = find_medoids(
                points, temp, possible_medoids, swap_count
            )  # temp is T (the temperature value)
        _, possible_medoids, loss = swap_temp(possible_medoids, loss, T_values)
        if medoids == medoids_p:
            same += 1
        else:
            same = 0
        medoids_p = medoids

    print("Number of swaps before convergence", swap_count)
    return possible_medoids[T], min(loss.values())
