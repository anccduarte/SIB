
import numpy as np
import sys
sys.path.append("../data")
sys.path.append("../io")
sys.path.append("../model_selection")
sys.path.append("../metrics")
sys.path.append("../statistics")
from csv_file import read_csv_file
from dataset import Dataset
from distances import euclidean_distance
from rmse import rmse
from split import train_test_split
from typing import Callable

class KNNRegressor:

	"""
	Implements the K-Nearest Neighbors classifier.
	Distances between test examples and some label can be computed using:
		- euclidean_distance: sqrt(SUM[(pi - qi)^2])
		- manhattan_distance: SUM[abs(pi - qi)]
	"""

	def __init__(self, k: int, distance: Callable):
		"""
		Implements the K-Nearest Neighbors classifier.
		Distances between test examples and some label can be computed using:
			- euclidean_distance: sqrt(SUM[(pi - qi)^2])
			- manhattan_distance: SUM[abs(pi - qi)]

		Parameters
		----------
		k: int
			Number of neighbors to be used
		distance: callable
			Function used to compute the distances

		Attributes
		----------
		fitted: bool
			Whether the model is already fitted
		dataset: Dataset
			A Dataset object (training data)
		"""
		# parameters
		if k < 1:
			raise ValueError("The value of 'k' must be greater than 0.")
		self.k = k
		self.distance = distance
		# attributes
		self.fitted = False
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

	print("EX - cpu")
	path_to_file = "../../../datasets/cpu/cpu.csv"
	cpu = read_csv_file(file=path_to_file, sep=",", features=True, label=True)
	cpu_trn, cpu_tst = train_test_split(dataset=cpu, test_size=0.3, random_state=12)
	knn = KNNRegressor(k=4, distance=euclidean_distance)
	knn.fit(cpu_trn)
	rmse = knn.score(cpu_tst)
	print(f"RMSE: {rmse:.4f}")

