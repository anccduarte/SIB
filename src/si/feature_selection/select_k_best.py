
import numpy as np
import sys
sys.path.append("../data")
sys.path.append("../statistics")
from dataset import Dataset
from f_classification import f_classification # f_regression (?)
from typing import Callable

class SelectKBest:

	"""
	Selects features according to the k highest scores.
	Feature ranking is performed by computing the scores of each feature using a scoring function:
		- f_classification: ANOVA F-value of each feature with examples grouped by label
		- f_regression (?)
	"""

	def __init__(self, score_func: Callable, k: int):
		"""
		Selects features according to the k highest scores.

		Parameters
		----------
		score_func: callable
			Function that takes a Dataset object and returns a tuple of arrays (F-scores and p-values)
		k: int
			Number of features to select

		Attributes
		----------
		F: np.ndarray
			The F-score(s) of feature(s)
		p: np.ndarray
			The p-value(s) of F-score(s)
		"""
		if k < 1:
			raise ValueError("The value of 'k' must be greater than 0.")
		self.score_func = score_func
		self.k = k
		self.F = None
		self.p = None

	def fit(self, dataset: Dataset) -> 'SelectKBest':
		"""
		Fits SelectKBest to compute the F-scores and p-values of the dataset's features.
		Returns self.

		Parameters
		----------
		dataset: Dataset
			A labeled Dataset object
		"""
		self.F, self.p = self.score_func(dataset)
		return self

	def transform(self, dataset: Dataset) -> Dataset:
		"""
		Transforms the dataset by selecting the k best features of the dataset. Returns a new Dataset
		object only containing the selected features.

		Parameters
		----------
		dataset: Dataset
			A labeled Dataset object
		"""
		idxs = np.argsort(self.F)[-self.k:]
		new_X = dataset.X[:,idxs]
		new_feats = np.array(dataset.features)[idxs]
		return Dataset(new_X, dataset.y, list(new_feats), dataset.label)

	def fit_transform(self, dataset: Dataset) -> Dataset:
		"""
		Fits SelectKBest and transforms the dataset by selecting the k best features. Returns a new 
		Dataset object only containing the selected features.

		Parameters
		----------
		dataset: Dataset
			A labeled Dataset object
		"""
		self.fit(dataset)
		return self.transform(dataset)


if __name__ == "__main__":
	X = np.array([[1,2,3,4],[3,6,5,1],[7,4,1,5],[1,3,2,9]])
	y = np.array([1,1,0,0])
	ds = Dataset(X,y)
	selector = SelectKBest(score_func=f_classification, k=2)
	new_ds = selector.fit_transform(ds)
	print(new_ds.X)

