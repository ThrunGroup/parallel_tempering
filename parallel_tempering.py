import random
import math
import kmedoids
import numpy as np
import argparse
import sklearn.metrics


def closest_dist(
        medoids: list,
        x: np.array,
) -> int:
    """
  Finds the smallest distance between x and the closest medoid

  :param medoids: list of all the medoids
  :param x: a point
  :return: the smallest Euclidean distance between the point and its closest medoid
  """

    dists = np.array([np.linalg.norm(i - x) for i in medoids])
    return np.min(dists)


def total_dist(
        points: list,
        medoids: list,
) -> int:
    """
  Adds up the closest_dist for each point with its closest medoid

  :param points: list of all the points (including the medoids)
  :param medoids: list of the medoids
  :return: the sum of smallest Euclidean distances for all points
  """

    closest_dists = np.array([closest_dist(medoids, i) for i in points])
    return np.sum(closest_dists)


def remove_array(
        list_arr : list,
        arr : np.array
) -> None:
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


def swap_medoids(
        medoids: list,
        points: list,
        T: float,
) -> list:
    """
  Returns the best state and its respective loss for which the total Euclidean distance is smallest

  :param medoids: list of the medoids
  :param T: represents the temperature, affecting the model's transition probability
  :param points: list of all the points (including the medoids)
  :return: the set of medoids after the swap with largest transition probability
  """
    non_medoids = []
    for i in points:
        if np.any(np.all(i == medoids, axis=1)):
            non_medoids.append(i)

    m = random.choice(medoids)
    n_m = random.choice(non_medoids)

    # new_state represents the set of medoids
    new_state = medoids[:]
    new_state.append(n_m)
    remove_array(new_state, m)

    # f(x') or f_new_state is just total_dist(points, state)
    initial_f = total_dist(points, medoids)
    f_new_state = total_dist(points, new_state)
    # make the swap with a probability proportional to the transition probability
    tp = float((math.e) ** (((-1 / T) * (initial_f - f_new_state))))
    if random.random() < tp:
        medoids = new_state
    total_loss = f_new_state

    return medoids, total_loss


def pypi_faster_pam(
        points: list,
        k: int,
) -> int:
    """
  Returns the overall loss using the in-built pypi package

  :param points: list of all the points (including the medoids)
  :param k: the number of medoids in a sample of points
  :return: the total loss
  """
    # uses the in-built Pypi FasterPAM package to find the overall loss
    diss = sklearn.metrics.pairwise.euclidean_distances(points)
    fp = kmedoids.fasterpam(diss, k)
    return fp.loss


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print overall loss comparison (default: True)",
    )


def check_equality(
        list1 : list,
        list2 : list
) -> bool:
    """
    Returns whether the lists (consisting of numpy arrays) are equal

    :param list1: the first list (consists of numpy ndarrays)
    :param list2: the second list (consists of numpy ndarrays)
    :return: the boolean value of their equality
    """
    size = len(list1)
    for i in range(size):
        if np.array_equal(list1[i], list2[i]) == False:
            return False
    return True


def main(
        points: list,
        T: float,
        k: int,
) -> int:
    """
  Returns k medoids for a set of points depending on a certain temperature

  :param points: list of all the points (including the medoids)
  :param T: represents the temperature, affecting the model's transition probability
  :param k: the number of medoids in a sample of points
  :return: the best state for medoids
  """
    args = parse_args()
    # medoids currently represents the initialization of choosing the medoids
    medoids = random.sample(points, k)

    # run chains for various values of T  (temperature)
    # T is geometrically defined (0, T, 2T, 4T, 8T, etc.)
    # continue swaps until convergence where the medoids prior to & after swapping are the same
    # medoids_p is the set of medoids prior to swapping
    # medoids_a is the set of medoids after swapping
    medoids_p = medoids  # for the initialization, medoids is the set prior to swapping
    swap_count = 0
    while True:
        medoids_a, pt_total_loss = swap_medoids(medoids_p, points, T)
        if check_equality(medoids_a, medoids_p):
            break
        medoids_p = medoids_a
        swap_count += 1

    pypi_loss = pypi_faster_pam(points, k)
    print("For the value of T", T)
    print("Loss using parallel tempering: ", pt_total_loss)
    print("Loss using the in-built Pypi package: ", pypi_loss)
    print("Number of swaps before convergence", swap_count)
    return medoids_a


if __name__ == "__main__":
    rand_points = []
    for i in range(1, 100):
        rand_points.append(np.array([random.randint(1, 1000), random.randint(1, 1000)]))

    T = 1000
    for i in range(3):
        T *= 2
        print(T)
        main(rand_points, T, k=5)
