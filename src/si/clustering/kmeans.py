
import numpy as np
import sys
PATHS = ["../data", "../statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from distances import euclidean_distance
from typing import Callable

class KMeans:

    """
    Implements the K-Means clustering algorithm.
    Distances can be computed using two distinct formulas:
        - euclidean_distance: sqrt(SUM[(pi - qi)^2])
        - manhattan_distance: SUM[abs(pi - qi)]
    """

    def __init__(self,
                 k: int = 6,
                 max_iter: int = 400, 
                 distance: Callable = euclidean_distance, 
                 seed: int = None):
        """
        Implements the K-Means clustering algorithm.
        Distances can be computed using two distinct formulas:
            - euclidean_distance: sqrt(SUM[(pi - qi)^2])
            - manhattan_distance: SUM[abs(pi - qi)]

        Parameters
        ----------
        k: int
            Number of clusters/centroids
        max_iter: int
            Maximum number of iterations for a single run
        distance: callable
            Function that computes distances
        seed: int
            Seed for the permutation generator used in centroid initialization

        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        centroids: np.ndarray
            An array containing the coordinates of the centroids
        labels: np.ndarray
            An array containing the clusters to which each sample belongs
        """
        # parameters
        if k < 2:
            raise ValueError("The value of 'k' must be greater than 1.")
        if max_iter < 1:
            raise ValueError("The value of 'max_iter' must be greater than 0.")
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.seed = seed
        # attributes
        self.fitted = False
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        Randomly generates the initial coordinates of the centroids.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        # returns a 1-dimensional array containing the numbers 0 through n_samples-1
        # (randomly distributed)
        perms = np.random.RandomState(seed=self.seed).permutation(dataset.shape()[0])
        # 1-dimenional array containing the first k numbers in perms
        seeds = perms[:self.k]
        # random initialization of the centroids (initially, each centroid corresponds to a sample)
        self.centroids = dataset.X[seeds]

    def _get_closest_centroid(self, sample: np.ndarray) -> int:
        """
        Returns the index of the closest centroid to a given sample.

        Parameters
        ----------
        sample: np.ndarray
            The sample to be assigned to a centroid
        """
        distances_to_centroids = self.distance(sample, self.centroids)
        closest_centroid = np.argmin(distances_to_centroids)
        return closest_centroid

    def fit(self, dataset: Dataset) -> "KMeans":
        """
        Fits KMeans by grouping all samples of a given dataset object in k clusters. To do so,
        repeatidly finds the coordinates of k centroids, assigning each sample of the dataset
        to the centroid it is closest to (Euclidean/Manhattan distance). It stops running when 
        a maximum number of iterations is hit or when convergence is declared (no changes in 
        sample assignment between two iterations of the algorithm). Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        # initialize centroids and labels
        self._init_centroids(dataset) # (k,)
        labels = np.zeros((dataset.shape()[0])) # (n_samples,)
        # main loop -> update centroids and labels
        i = 0
        converged = False
        while i < self.max_iter and not converged:
            # get closest centroid to each sample of the dataset (along each sample -> axis=1)
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
            # if labels == new_labels break out of while loop (label assignment converged)
            if (labels == new_labels).all():
                converged = True
            else:
                # re-compute the new centroids:
                # 1. get samples at centroid j
                # 2. get the mean values of the columns of those samples (computed along axis=0)
                centroids = [np.mean(dataset.X[new_labels==j], axis=0) for j in range(self.k)]
                self.centroids = np.array(centroids)
                # in order to compare labels and new_labels in the next iteration
                labels = new_labels.copy()
                i += 1
        # update attributes
        self.fitted = True
        self.labels = new_labels # only assign in 'predict' ???
        return self

    def _get_distances_to_centroids(self, sample: np.ndarray) -> np.ndarray:
        """
        Computes and returns the distances between a given sample and all centroids.

        Parameters
        ----------
        sample: np.ndarray
            The sample whose distances to the centroids are to be computed
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transforms the dataset by computing the distances of all samples to the centroids.
        Returns an array of shape (n_samples, k), where each row represents the distances of
        each sample to all centroids.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        if not self.fitted:
            raise Warning("Fit 'KMeans' before calling 'transform'.")
        distances_to_centroids = np.apply_along_axis(self._get_distances_to_centroids, axis=1, arr=dataset.X)
        return distances_to_centroids

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Fits KMeans and transforms the dataset by computing the distances of all samples to
        the centroids. Returns an array of shape (n_samples, k), where each row represents the
        distances of each sample to all centroids.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset):
        """
        Predicts the cluster to which all samples of the dataset belong. Returns a 1-dimensional
        vector containing the predictions.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        if not self.fitted:
            raise Warning("Fit 'KMeans' before calling 'predict'.")
        else:
            self.labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
            return self.labels

    def fit_predict(self, dataset: Dataset):
        """
        Fits KMeans and predicts the cluster to which all samples of the dataset belong. Returns 
        a 1-dimensional vector containing the predictions.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.predict(dataset)


if __name__ == "__main__":

    sys.path.append("../io")
    from csv_file import read_csv_file
    from distances import manhattan_distance

    print("EX1")
    ds1 = Dataset.from_random(n_examples=10, n_features=10, label=False, seed=0)
    km1 = KMeans(k=2)
    # print(km1.fit_transform(ds1))
    print(km1.fit_predict(ds1))
    
    print("\nEX2")
    ds2 = Dataset.from_random(n_examples=20, n_features=20, label=False, seed=1)
    km2 = KMeans(k=4, max_iter=200, distance=manhattan_distance, seed=3)
    # print(km2.fit_transform(ds2))
    print(km2.fit_predict(ds2))

    print("\nEX3 - iris")
    path = "../../../datasets/iris/iris.csv"
    iris = read_csv_file(path, sep=",", features=True, label=True)
    km3 = KMeans(k=3, max_iter=1000, distance=euclidean_distance, seed=0)
    print(km3.fit_predict(iris))
    
