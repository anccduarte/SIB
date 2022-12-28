
import numpy as np
import sys
PATHS = ["../data", "../statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from distances import euclidean_distance
from typing import Callable

class KMeans:

    """
    Implements the K-Means clustering algorithm. Distances between data points can be computed
    using one of two distinct formulas:
        - euclidean_distance: sqrt(SUM[(pi - qi)^2])
        - manhattan_distance: SUM[abs(pi - qi)]
    """

    def __init__(self,
                 k: int = 5,
                 num_init: int = 10,
                 max_iter: int = 1000,
                 tolerance: int = 0,
                 distance: Callable = euclidean_distance, 
                 seed: int = None):
        """
        Implements the K-Means clustering algorithm. Distances between data points can be computed
        using one of two distinct formulas:
            - euclidean_distance: sqrt(SUM[(pi - qi)^2])
            - manhattan_distance: SUM[abs(pi - qi)]

        Parameters
        ----------
        k: int (default=5)
            Number of clusters/centroids
        num_init: int (default=10)
            Number of times the K-Means algorithm is run
        max_iter: int (default=1000)
            Maximum number of iterations for a single run
        tolerance: int (default=0)
            The required minimum number of changes in label assignment between iterations of a single
            run in order to declare convergence
        distance: callable (default=euclidean_distance)
            Function that computes distances between data points
        seed: int (default=None)
            Seed for the permutation generator used in centroid initialization (if 'int', after each
            iteration of the algorithm, it is updated by doing seed := seed + 1)

        Attributes
        ----------
        fitted: bool
            Whether 'KMeans' is already fitted
        centroids: np.ndarray
            An array containing the coordinates of the centroids (it is continuously updated during
            iterations of the algorithm; after fitting, its value is updated to the centroid coordinates
            found during the iteration which produced the smallest value of inertia)
        inertia: float
            Sum squared distances of each sample to the respective cluster (the minimum value found
            during <self.num_init> iterations)
        labels: np.ndarray
            An array containing the clusters to which each sample belongs
        """
        # check parameters
        self._check_init(k, num_init, max_iter, tolerance)
        # parameters
        self.k = k
        self.num_init = num_init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.distance = distance
        self.seed = seed
        # attributes
        self.fitted = False
        self.centroids = None
        self.inertia = None
        self.labels = None

    @staticmethod
    def _check_init(k: int, num_init: int, max_iter: int, tolerance: int):
        """
        Checks the values of numeric parameters.

        Parameters
        ----------
        k: int
            Number of clusters/centroids
        num_init: int
            Number of times the K-Means algorithm is run
        max_iter: int
            Maximum number of iterations for a single run
        tolerance: int
            The required minimum number of changes in label assignment between iterations of a single
            run in order to declare convergence
        """
        if k < 2:
            raise ValueError("The value of 'k' must be greater than 1.")
        if num_init < 1:
            raise ValueError("The value of 'num_init' must be grater than 0.")
        if max_iter < 1:
            raise ValueError("The value of 'max_iter' must be greater than 0.")
        if tolerance < 0:
            raise ValueError("The value of 'tolerance' must be a non-negative integer.")

    def _init_centroids(self, dataset: Dataset):
        """
        Randomly generates the initial centroid coordinates.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be clustered)
        """
        # returns a 1-dimensional array containing the numbers 0 through n_samples-1
        # (randomly distributed)
        perms = np.random.RandomState(seed=self.seed).permutation(dataset.shape()[0])
        # 1-dimenional array containing the first k numbers in perms
        samples = perms[:self.k]
        # random initialization of the centroids (initially, each centroid corresponds to a sample)
        self.centroids = dataset.X[samples]
        # reassign seed so that new centroids are chosen in the next iteration
        if self.seed is not None:
            self.seed += 1

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

    def _get_inertia(self, dataset: Dataset) -> float:
        """
        Computes and returns the unweighted sum of squared distances of samples to their respective
        centroids. Lower inertia values are sign of a better distribution of samples between cluster
        centers (centroids).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be clustered)
        """
        # get labels of each sample
        labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
        # get clusters
        clusters = [dataset.X[labels == i] for i in range(self.k)]
        # determine squared distances for each cluster
        s_dists = [np.square(self.distance(self.centroids[i], clusters[i])) for i in range(self.k)]
        # compute and return inertia
        inertia = sum([np.sum(d) for d in s_dists])
        return inertia

    def _run_kmeans(self, dataset: Dataset) -> None:
        """
        Runs an iteration of the K-Means algorithm. It repeatidly reassigns labels, and computes
        new centroid coordinates by calculating the mean value of each cluster. It stops running when
        a maximum number of iterations is hit or when convergence is declared (number of changes in
        label assignment less than or equal to <self.tolerance> between two iterations).
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (ontaining the data to be clustered)
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
            # if num_changes <= tolerance, break out of while loop (label assignment converged)
            if np.sum(labels != new_labels) <= self.tolerance:
                converged = True
            else:
                # compute the coordinates of the new centroids:
                # 1. get samples at centroid j
                # 2. get the mean values of the columns of those samples (computed along axis=0)
                centroids = [np.mean(dataset.X[new_labels==j], axis=0) for j in range(self.k)]
                self.centroids = np.array(centroids)
                # in order to compare labels and new_labels in the next iteration
                labels = new_labels.copy()
                i += 1

    def fit(self, dataset: Dataset) -> "KMeans":
        """
        Fits KMeans by determining the coordinates of <self.k> centroids <self.num_init> times.
        It chooses the centroid coordinates which produce the smallest value of inertia. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be clustered)
        """
        best_inertia = None
        best_centroids = None
        # main loop -> run <self.num_init> kmeans iterations
        for _ in range(self.num_init):
            # run kmeans iteration
            try:
                self._run_kmeans(dataset)
            # cases where some cluster is empty after an iteration
            except RuntimeWarning:
                print("Warning: At least one cluster is empty. Ignoring iteration...")
            # get inertia and check if it is the best so far
            else:
                inertia = self._get_inertia(dataset)
                if (best_inertia is None) or (inertia < best_inertia):
                    best_inertia = inertia
                    best_centroids = self.centroids.copy()
        # update attributes and return
        self.fitted = True
        self.inertia = best_inertia
        self.centroids = best_centroids.copy()
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
        each sample to all k centroids.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be clustered)
        """
        if not self.fitted:
            raise Warning("Fit 'KMeans' before calling 'transform'.")
        distances = np.apply_along_axis(self._get_distances_to_centroids, axis=1, arr=dataset.X)
        return distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Fits KMeans and transforms the dataset by computing the distances of all samples to
        the centroids. Returns an array of shape (n_samples, k), where each row represents the
        distances of each sample to all k centroids.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be clustered)
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
            A Dataset object (containing the data to be clustered)
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
            A Dataset object (containing the data to be clustered)
        """
        self.fit(dataset)
        return self.predict(dataset)


if __name__ == "__main__":

    sys.path.append("../io")
    from csv_file import read_csv_file
    from distances import manhattan_distance

    print("EX1")
    ds1 = Dataset.from_random(n_examples=10, n_features=10, label=False, seed=0)
    km1 = KMeans(k=2, max_iter=400)
    # print(km1.fit_transform(ds1))
    print(km1.fit_predict(ds1))
    print(f"Inertia: {km1.inertia:.4f}")
    
    print("\nEX2")
    ds2 = Dataset.from_random(n_examples=20, n_features=20, label=False, seed=1)
    km2 = KMeans(k=4, max_iter=200, distance=manhattan_distance, seed=3)
    # print(km2.fit_transform(ds2))
    print(km2.fit_predict(ds2))
    print(f"Inertia: {km2.inertia:.4f}")

    tol = 2
    print(f"\nEX3 - iris (tolerance={tol})")
    path = "../../../datasets/iris/iris.csv"
    iris = read_csv_file(path, sep=",", features=True, label=True)
    km3 = KMeans(k=3, tolerance=tol, seed=0)
    print(km3.fit_predict(iris))
    print(f"Inertia: {km3.inertia:.4f}")
    
