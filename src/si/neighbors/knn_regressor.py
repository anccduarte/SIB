
import numpy as np
import sys
PATHS = ["../data", "../metrics", "../statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from distances import euclidean_distance
from rmse import rmse
from typing import Callable

class KNNRegressor:

    """
    Implements the K-Nearest Neighbors regressor.
    Distances between test examples and some label can be computed using:
        - euclidean_distance: sqrt(SUM[(pi - qi)^2])
        - manhattan_distance: SUM[abs(pi - qi)]
    """

    def __init__(self, k: int = 4, weighted: bool = False, distance: Callable = euclidean_distance):
        """
        Implements the K-Nearest Neighbors regressor.
        Distances between test examples and some label can be computed using:
            - euclidean_distance: sqrt(SUM[(pi - qi)^2])
            - manhattan_distance: SUM[abs(pi - qi)]

        Parameters
        ----------
        k: int (default=4)
            Number of neighbors to be used
        weighted: bool (default=False)
            Whether to weight closest neighbors when predicting labels
        distance: callable (default=euclidean_distance)
            Function used to compute the distances

        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        weights_vector: np.ndarray
            The weights to give to each closest neighbor when predicting labels (only applicable
            when 'weights' is True)
        dataset: Dataset
            A Dataset object (training data)
        """
        # parameters
        if k < 1:
            raise ValueError("The value of 'k' must be greater than 0.")
        self.k = k
        self.weighted = weighted
        self.distance = distance
        # attributes
        self.fitted = False
        if self.weighted:
            self.weights_vector = np.arange(self.k,0,-1)
        self.dataset = None

    def fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Stores the training dataset. Returns self.
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (training data)
        """
        self.dataset = dataset
        self.fitted = True
        return self

    def _get_closest_labels_mean(self, sample: np.ndarray) -> float:
        """
        Returns the mean value of the closest labels to the sample.

        Parameters
        ----------
        sample: np.ndarray
            The sample to be labeled
        """
        # calculate distances
        distances = self.distance(sample, self.dataset.X)
        # determine indices of the closest neighbors
        label_indices = np.argsort(distances)[:self.k]
        # get the values at the previous indices
        label_vals = self.dataset.y[label_indices]
        # tranform labels vector to account for weights (if applicable)
        if self.weighted:
            label_vals = np.repeat(label_vals, self.weights_vector)
        # compute the mean value and return it
        return np.mean(label_vals)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the classes of the dataset given as input.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (testing data)
        """
        if not self.fitted:
            raise Warning("Fit 'KNNRegressor' before calling 'predict'.")
        return np.apply_along_axis(self._get_closest_labels_mean, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates and returns the error between the predicted and true classes. To compute
        the error, it uses the RMSE: sqrt((SUM[(ti - pi)^2]) / N).
    
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (testing data)
        """
        if not self.fitted:
            raise Warning("Fit 'KNNRegressor' before calling 'score'.")
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../model_selection"]
    sys.path.extend(TEST_PATHS)
    from csv_file import read_csv_file
    from split import train_test_split

    print("EX - cpu")
    path_to_file = "../../../datasets/cpu/cpu.csv"
    cpu = read_csv_file(file=path_to_file, sep=",", features=True, label=True)
    cpu_trn, cpu_tst = train_test_split(dataset=cpu, test_size=0.3, random_state=1)
    knn = KNNRegressor(k=4, weighted=True, distance=euclidean_distance)
    knn.fit(cpu_trn)
    rmse = knn.score(cpu_tst)
    print(f"RMSE: {rmse:.4f}")

