import random
import math
import kmedoids
import numpy as np
import argparse
import copy
import cProfile
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def pypi_faster_pam(points: list, k: int,) -> int:
    """
    Returns the overall loss using the in-built pypi package
    :param points: list of all the points (including the medoids)
    :param k: the number of medoids in a sample of points
    :return: the total loss
    """
    # uses the Pypi FasterPAM kmedoids package to find the overall loss
    diss = sklearn.metrics.pairwise.euclidean_distances(points)
    fp = kmedoids.fasterpam(diss, k)
    return fp.loss


def total_dist(points: list, medoids: np.array,) -> int:
    """
    Adds up the closest_dist for each point with its closest medoid
    :param points: list of all the points (including the medoids)
    :param medoids: numpy array of the medoids
    :return: the sum of smallest Euclidean distances for all points
    """
    min_dists = [np.min(np.linalg.norm(medoids - i, axis=1), axis=0) for i in points]
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


def swap_medoids(
    medoids: np.array, points: list, T: float, swap_count: int, initial_loss: float
) -> np.array:
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
    tp = min(
        1,
        np.float128(
            (math.e) ** (((1 / (initial_loss * T)) * (initial_f - f_new_state)))
        ),
    )
    print(initial_f, f_new_state)
    # print("tp is", np.float128((math.e) ** (((-1 / (initial_loss * T)) * (initial_f - f_new_state)))))
    if random.uniform(0, 1) < tp:
        medoids = copy.deepcopy(new_state)
        swap_count += 1
        total_loss = f_new_state
    else:
        total_loss = initial_f

    return medoids, total_loss, swap_count


def find_medoids(
    points: list, T: float, possible_medoids: dict, swap_count: int, initial_loss: float
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
    medoids_a, pt_total_loss, swap_count = swap_medoids(
        medoids_p, points, T, swap_count, initial_loss
    )
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

    # if the loss with higher temperature is less than the loss with lower temperature, we swap the set of medoids and the loss
    for temp in T_values:
        temp_idx = T_values.index(temp)
        for greater_temp in T_values[temp_idx + 1 :]:
            if loss[temp] > loss[greater_temp]:
                possible_medoids[temp], possible_medoids[greater_temp] = (
                    possible_medoids[greater_temp],
                    possible_medoids[temp],
                )
                loss[temp], loss[greater_temp] = loss[greater_temp], loss[temp]

    lowest_loss = min(loss.values())
    return (lowest_loss, possible_medoids, loss)


def build_init(points: list, medoids: np.array):
    """
    Returns a medoid according to the current set of k medoids giving the smallest possible loss overall
    :param points: list of all the points (including the medoids)
    :param medoids: array of the medoids
    :return: the medoid to be added giving the lowest loss overall
    """
    losses = np.array([total_dist(points, medoids + [i]) for i in points])
    medoid = [i for i in range(len(losses)) if losses[i] == np.min(losses)][0]
    return points[medoid]


def pt_plot(history: dict, T_values: list, start: int, end: int, pypi_loss: int):
    """
    Creates a parallel tempering plot 
    :param history: stores all previous losses for each temperature
    :param T_values: a list of all temperatures for each running chain
    :param start: the beginning of the range to display swaps from
    :param end: the end of the range mentioned above
    :param pypi_loss: the loss from the pypi package of kmedoids
    :return: None
    """
    plt.figure(figsize=[16, 8])
    for i in T_values:
        x = history[i]
        history[i] = x[start:end]  # include range (how many swaps) e.g. [10:100]
        xs = list(range(len(history[i])))
        ys = history[i]
        plt.plot(xs, ys, linewidth=2, label=f"{i}")

    plt.axhline(y = pypi_loss, color = 'r', linestyle = '-.')
    plt.legend()
    plt.show()
    plt.close()
    return None


def main(points: list, T: float, k: int, conv_condition: int, num_temp: int) -> int:
    """
    Returns k medoids for a set of points depending on a certain temperature
    :param points: list of all the points (including the medoids)
    :param T: represents the temperature, affecting the model's= transition probability
    :param k: the number of medoids in a sample of points
    :param num_temp: the number of temperature values
    :return: the best state for medoids and its corresponding loss
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, default=True,
        help="Whether to print overall loss comparison (default: True)")
    parser.add_argument("--iterations", type=int, default=-1, help="which part of the parallel tempering plot to display")

    args = parser.parse_args()

    # parameter keeps track of the number of swaps
    swap_count = 0

    # medoids currently represents the initialization of choosing the medoids
    # the seeding process uses the BUILD step of PAM
    medoids = []
    for i in range(k):
        medoids.append(build_init(points, medoids))

    print("init", total_dist(points, medoids))

    # scale the transition probability with a factor of the initial loss
    initial_loss = total_dist(points, medoids)

    # dictionaries containing the set of medoids and its respective overall loss for each value of T
    possible_medoids = {}
    loss = {}
    history = {}
    T_values = [T * (1.3 ** i) for i in range(num_temp)]
    # initially, the medoids for each temperature chain is just the random sample generated above
    for i in T_values:
        possible_medoids[i] = medoids
        history[i] = []

    same = 0
    medoids_p = medoids
    while same < conv_condition:
        for temp in T_values:
            possible_medoids[temp], loss[temp], swap_count = find_medoids(
                points, temp, possible_medoids, swap_count, initial_loss
            )  # temp is T (the temperature value)
            history[temp].append(loss[temp])
        _, possible_medoids, loss = swap_temp(possible_medoids, loss, T_values)
        if medoids == medoids_p:
            same += 1
        else:
            same = 0
        medoids_p = medoids

    pypi_loss = pypi_faster_pam(points, k)
    print("For the value of T", T)
    print("Loss using parallel tempering: ", min(loss.values()))
    print("Loss using the in-built Pypi package: ", pypi_loss)
    print("Number of swaps before convergence", swap_count)
    print(loss)

    # include specifc range of swaps (start to end) and the temperature values to plot
    end = args.iterations
    pt_plot(history, T_values, start=0, end=end, pypi_loss = pypi_loss)
    return possible_medoids[T]


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    rand_points = list(np.random.randint(1, 1000, size=(100, 2)))

    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    mnist = X[:10000]
    T = 0.001
    main(rand_points, T, k=5, conv_condition=800, num_temp=10)
    # main(mnist, T, k=5, conv_condition=500, num_temp=10)
    # cProfile.runctx('main(rand_points, T, k=5, conv_condition=300, num_temp=10)', globals(),locals())

# in the command line, run the program as e.g. "parallel_tempering.py --verbose True --iterations 100"
