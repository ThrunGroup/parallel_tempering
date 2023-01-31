from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X[:10000]
