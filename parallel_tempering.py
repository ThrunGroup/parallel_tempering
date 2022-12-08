#!pip install kmedoids
#^this must be installed via terminal

import random
import math
import kmedoids
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

rand_points = []
for i in range(1, 100):
  rand_points.append((random.randint(1,1000), random.randint(1,1000)))

def euc_dist(
    p : tuple,
    q : tuple,
) -> int:
  """
  Finds the euclidean distance between points p and q

  :param p: the first point
  :param q: the second point
  :return: the Euclidean distance between the points
  """
  # f(x) is the measure of these states (in terms of euclidean distance)
  dist = ((q[0] - p[0])**2 + (q[1] - p[1])**2)**0.5
  return dist

def closest_dist(
    medoids : list,
    x : tuple,
) -> int:
  """
  Finds the smallest distance between x and the closest medoid

  :param medoids: list of all the medoids
  :param x: a point
  :return: the smallest Euclidean distance between the point and its closest medoid
  """
  min = 2**32 - 1
  for i in medoids:
    if euc_dist(i, x) < min:
      min = euc_dist(i, x)
  return min

def total_dist(
    points : list, 
    medoids : list,
) -> int:
  """
  Adds up the closest_dist for each point with its closest medoid

  :param points: list of all the points (including the medoids)
  :param medoids: list of the medoids
  :return: the sum of smallest Euclidean distances for all points
  """
  total = 0
  for i in points:
    total += closest_dist(medoids, i)
  return total

def possible_states(
    points : list, 
    medoids : list,
) -> None:
  """
  Returns the possible states for medoids (potential combinations of medoids 
  after performing swaps)

  :param points: list of all the points (including the medoids)
  :param medoids: list of the medoids
  :return: the list representing the possible states of medoids
  """
  # states represent possible states for the medoids
  # which result from swapping a (medoid, non-medoid) pair
  states = []
  non_medoids = []
  for i in points:
    if i not in medoids:
      non_medoids.append(i)
  medoids_c = medoids[:]
  for i in medoids_c:
    for j in non_medoids:
      medoids.append(j)
      medoids.remove(i) 
      states.append(medoids)
      medoids = medoids_c[:]
  return states

def swap_medoids(
    states : list, 
    medoids : list, 
    points : list, 
    T : int,
) -> list:
  """
  Returns the best state for which the total Euclidean distance is smallest

  :param states: list representing the list of possible medoids
  :param medoids: list of the medoids
  :param T: represents the temperature, affecting the model's transition probability
  :param points: list of all the points (including the medoids)
  :return: the set of medoids after the swap with largest transition probability
  """
  best_state = states[0]
  tp = float(0) # transition probability
  for state in states:
    # state represents the set of medoids
    # f(x') or f_new_state is just total_dist(points, state)
    initial_f = total_dist(points, medoids)
    f_new_state = total_dist(points, state)
    print("init_f", initial_f, "f_new_state", f_new_state)
    print("exp", (float(math.e)**(((-1/T) * (initial_f - f_new_state)))))
    if (float((math.e)**(((-1/T) * (initial_f - f_new_state)))) > tp):
      tp = (math.e)**((-1/T) * (initial_f - f_new_state))
      print(T, (math.e)**((-1/T) * (initial_f - f_new_state)))
      best_state = state
    total_loss = f_new_state

  # make the swap which corresponds to the largest transition probability
  medoids = best_state
  return medoids, total_loss

def decision(probability):
    return random.random() < probability

def swap_medoids_prob(
    states : list, 
    medoids : list, 
    points : list, 
    T : int,
) -> list:
  """
  Returns the best state for which the total Euclidean distance is smallest

  :param states: list representing the list of possible medoids
  :param medoids: list of the medoids
  :param T: represents the temperature, affecting the model's transition probability
  :param points: list of all the points (including the medoids)
  :return: the set of medoids after the swap with largest transition probability
  """
  best_state = states[0]
  tp = float(0) # transition probability
  for state in states:
    # state represents the set of medoids
    # f(x') or f_new_state is just total_dist(points, state)
    initial_f = total_dist(points, medoids)
    f_new_state = total_dist(points, state)
    prob = float((math.e)**(((-1/T) * (initial_f - f_new_state))))
    if decision(prob):
      best_state = state
    total_loss = f_new_state

  # make the swap with a probability proportional to the transition probability
  medoids = best_state
  return medoids, total_loss


def pypi_faster_pam(
    points : list, 
    k : int,
) -> int:
  k = 10
  diss = euclidean_distances(points)
  fp = kmedoids.fasterpam(diss, k)
  return fp.loss

def main(
    points : list, 
    T : int, 
    k : int,
) -> int:
  """
  Returns k medoids for a set of points depending on a certain temperature

  :param points: list of all the points (including the medoids)
  :param T: represents the temperature, affecting the model's transition probability
  :param k: the number of medoids in a sample of points
  :return: the best state for medoids
  """
  # medoids currently represents the initialization of choosing the medoids
  medoids = random.sample(points, k) 
  
  # run chains for various values of T  (temperature)
  # T is geometrically defined (0, T, 2T, 4T, 8T, etc.)
  states = possible_states(points, medoids)
  medoids, pt_total_loss = swap_medoids_prob(states, medoids, points, T)
  pypi_loss = pypi_faster_pam(points, k)
  print("For the value of T,", T)
  print("Loss using parallel tempering: ", pt_total_loss)
  print("Loss using the in-built Pypi package: ", pypi_loss)
  return medoids

T = 0.001 #initial value of T
for i in range(5):
    T *= 2
    print(T)
    main(rand_points, T = 0.001, k = 5)