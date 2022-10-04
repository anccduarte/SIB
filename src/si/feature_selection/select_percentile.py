
import numpy as np
import sys
sys.path.append("../data")
sys.path.append("../statistics")
from dataset import Dataset
from f_classification import f_classification # f_regression (?)
from typing import Callable

class SelectPercentile:

	"""
	Selects the best features according to a given percentile.
	Feature ranking is performed by computing the scores of each feature using a scoring function:
		- f_classification: ANOVA F-value of each feature with examples grouped by label
		- f_regression (?)
	"""

	def __init__(self, score_func: Callable, percentile: float):
		"""
		Selects the best features according to a given percentile.

		Parameters
		----------
		score_func: callable
			Function that takes a Dataset object and returns a tuple of arrays (F-scores and p-values)
		percentile: float
			The percentage of features to be selected

		Attributes
		----------
		F: np.ndarray
			The F-score(s) of feature(s)
		p: np.ndarray
			The p-value(s) of F-score(s)
		"""
		if percentile < 0 or percentile > 1:
			raise ValueError("The value of 'percentile' must be in (0,1).")
		self.score_func = score_func
		self.percentile = percentile
		self.F = None
		self.p = None

	def fit(self, dataset: Dataset) -> 'SelectPercentile':
		"""
		Fits SelectPercentile by computing the F-scores and p-values of the dataset's features.
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
		Transforms the dataset by selecting the best features according to a given percentile.
		Returns a new Dataset object only containing the selected features.

		Parameters
		----------
		dataset: Dataset
			A labeled Dataset object
		"""
		n_feats = round(len(dataset.features)*self.percentile)
		idxs = np.argsort(self.F)[-n_feats:]
		new_X = dataset.X[:,idxs]
		new_fits = np.array(dataset.features)[idxs]
		return Dataset(new_X, dataset.y, list(new_fits), dataset.label)

	def fit_transform(self, dataset: Dataset) -> Dataset:
		"""
		Fits SelectPercentile and transforms the dataset by selecting the best features according
		to a given percentile. Returns a new Dataset object only containing the selected features.

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
	selector = SelectPercentile(score_func=f_classification, percentile=0.4)
	new_ds = selector.fit_transform(ds)
	print(new_ds.X)

