from parallel_tempering import main
import kmedoids
import numpy
import time
import random
from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances

rand_points = list(np.random.randint(1, 1000, size=(100, 2)))
X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:10000]
diss = euclidean_distances(X)
fp = kmedoids.fasterpam(diss, 100)
print("Loss with FasterPAM:", fp.loss)

# either dataset = list(X) for MNIST dataset or rand_points for smaller dataset
dataset = list(X)
T = [2, 3, 4, 5, 10]
k = 5
conv = [100, 250, 300, 500, 600]
num = [10, 20, 40, 50, 100]

for i in T:
    for j in conv:
        for n in num:
            print(f'T is {i}, conv is {j}, num_temp is {n}.')
            _, loss = main(dataset, T = i, k = k, conv_condition = j, num_temp = n)
            print("Loss using parallel tempering is ", loss)
