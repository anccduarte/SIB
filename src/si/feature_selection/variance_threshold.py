
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset
from typing import Union

class VarianceThreshold:

	"""
	Selects features according to a given variance threshold.
	Variance = SUM[(xi - xm)^2] / (n - 1) -> applied to each feature
	"""

	def __init__(self, threshold: Union[int, float]):
		"""
		Selects features according to a given variance threshold.
		Variance = SUM[(xi - xm)^2] / (n - 1) -> applied to each feature

		Parameters
		----------
		threshold: int, float
			The variance threshold

		Attributes
		----------
		variance: np.ndarray
			Array containing the variance of each feature of the dataset
		"""
		if threshold < 1:
			raise ValueError("The value of 'threshold' must be greater than 0.")
		self.threshold = threshold
		self.variance = None

	def fit(self, dataset: Dataset) -> 'VarianceThreshold':
		"""
		Fits VarianceThreshold to compute the variances of the dataset's features.
		Returns self.

		Parameters
		----------
		dataset: Dataset
			A Dataset object
		"""
		self.variance = dataset.get_variance()
		return self

	def transform(self, dataset: Dataset) -> Dataset:
		"""
		Transforms the dataset by selecting the features according to the variance threshold.
		Returns a new Dataset object only containing the selected features.

		Parameters
		----------
		dataset: Dataset
			A labeled Dataset object
		"""
		mask = self.variance > self.threshold
		new_X = dataset.X[:,mask]
		features = np.array(dataset.features)[mask]
		return Dataset(new_X, dataset.y, list(features), dataset.label)

	def fit_transform(self, dataset: Dataset) -> Dataset:
		"""
		Fits VarianceThreshold and transforms the dataset by selecting the features according to
		the variance threshold. Returns a new Dataset object containing the selected features.

		Parameters
		----------
		dataset: Dataset
			A Dataset object
		"""
		self.fit(dataset)
		return self.transform(dataset)


if __name__ == "__main__":
	# only column 1 has variance greater than 3
	X = np.array([[1,4,3],[7,5,6],[9,8,2]])
	y = np.array([10,11,12])
	ds = Dataset(X,y)
	selector = VarianceThreshold(3)
	new_ds = selector.fit_transform(ds)
	print(new_ds.X)

