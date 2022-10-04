
from typing import Tuple
import numpy as np
import pandas as pd

class Dataset:

	"""
	Constructs a tabular dataset for machine learning.
	"""
	
	def __init__(self, X: np.ndarray, y: np.ndarray = None, features: list = None, label: str = None):
		"""
		Constructs a tabular dataset for machine learning.

		Parameters
		----------
		X: np.ndarray
			The matrix containing the dataset's features
		y: np.ndarray
			The label vector
		features: list
			The names of the features
		label: str
			The name of the label
		"""
		if y is not None:
			if X.shape[0] != y.size:
				raise ValueError("The number of examples in 'X' must be equal to the size of 'y'.")
		self.X = X
		self.y = y
		self.features = [f"feat{i+1}" for i in range(X.shape[1])] if features is None else features
		self.label = "label" if (y is not None and label is None) else label

	def shape(self) -> Tuple[int, int]:
		"""
		Returns a two-element tuple consisting of the dataset's dimensions.
		"""
		return self.X.shape

	def has_label(self) -> bool:
		"""
		Returns a boolean value representative of the presence of a label.
		"""
		return True if self.y is not None else False

	def get_classes(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the unique classes of the dataset.
		"""
		if self.y is None:
			raise ValueError("The parameter 'y' was set to 'None'.")
		return np.unique(self.y)
		#return list(set(self.y))

	def get_mean(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the mean of each feature.
		"""
		return self.X.mean(axis=0)

	def get_variance(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the variance of each feature.
		"""
		return self.X.var(axis=0)

	def get_median(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the median of each feature.
		"""
		return np.median(self.X, axis=0)

	def get_min(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the minimum value of each feature.
		"""
		return self.X.min(axis=0)

	def get_max(self) -> np.ndarray:
		"""
		Returns an np.ndarray containing the maximum value of each feature.
		"""
		return self.X.max(axis=0)

	def summary(self) -> pd.DataFrame:
		"""
		Returns a pd.DataFrame containing some descriptive metrics (mean, variance, median,
		minimum value and maximum value) of each feature.
		"""
		df = pd.DataFrame({
			"Mean": self.get_mean(),
			"Variance": self.get_variance(),
			"Median": self.get_median(),
			"Min": self.get_min(),
			"Max": self.get_max()
			})
		return df

	def remove_nan(self):
		"""
		Removes examples which contain missing values (NaN).
		"""
		idx = np.isnan(self.X).any(axis=1)
		self.X = self.X[~idx]
		if self.y is not None:
			self.y = self.y[~idx]

	def fill_nan(self, fill: str):
		"""
		Replaces all dataset's missing values (NaN) by the mean/median of the respective column.
		Allowed values for 'fill' are:
			- 'mean': calls np.nanmean on the matrix containing the features
			- 'median': calls np.nanmedian on the matrix containig the features

		Parameters
		----------
		fill: str
			The string description of the value by which missing values are replaced
		"""
		vals = {"mean": np.nanmean, "median": np.nanmedian}
		self.X = np.nan_to_num(self.X, nan=vals[fill](self.X,axis=0))


if __name__ == "__main__":

	print("EX1")
	X1 = np.array([[1,2,3],[4,5,6]])
	y1 = np.array([1,2])
	f1 = ["A","B"]
	l1 = "y"
	ds1 = Dataset(X1, y1, f1, l1)
	print(f"shape: {ds1.shape()}")
	print(f"has_label: {ds1.has_label()}")
	print(f"classes: {ds1.get_classes()}")
	print(ds1.summary())

	print("\nEX2")
	X2 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
	y2 = np.array([7,8,9])
	ds2 = Dataset(X2, y2)
	print(f"shape (before removing NaNs): {ds2.shape()}")
	ds2.remove_nan()
	print(f"shape (after removing NaNs): {ds2.shape()}")
	print(ds2.summary())

	print("\nEX3")
	X3 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
	y3 = np.array([7,8,9])
	ds3 = Dataset(X3, y3)
	print(f"shape (before filling NaNs): {ds3.shape()}")
	ds3.fill_nan("mean")
	print(f"shape (after filling NaNs): {ds3.shape()}")
	print(ds3.summary())

