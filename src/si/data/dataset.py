
import numpy as np
import pandas as pd

class Dataset:

	def __init__(self, X, y=None, features=None, label=None):
		self.X = X
		self.y = y
		self.features = features
		self.label = label

	def shape(self):
		return self.X.shape

	def has_label(self):
		return True if self.y is not None else False

	def get_classes(self):
		return np.unique(self.y)
		# return list(set(self.y))

	def get_mean(self):
		return self.X.mean(axis=0)

	def get_variance(self):
		return self.X.var(axis=0)

	def get_median(self):
		return np.median(self.X, axis=0)

	def get_min(self):
		return self.X.min(axis=0)

	def get_max(self):
		return self.X.max(axis=0)

	def summary(self):
		df = pd.DataFrame({"Mean": self.get_mean(),
						   "Variance": self.get_variance(),
						   "Median": self.get_median(),
						   "Min": self.get_min(),
						   "Max": self.get_max()})
		return df

if __name__ == "__main__":
	x1 = np.array([[1,2,3],[4,5,6]])
	y1 = np.array([1,2])
	f1 = ["A","B"]
	l1 = "y"
	ds1 = Dataset(x1, y1, f1, l1)
	print(f"shape: {ds1.shape()}")
	print(f"has_label: {ds1.has_label()}")
	print(f"classes: {ds1.get_classes()}")
	print(f"means: {ds1.get_mean()}")
	print(f"variances: {ds1.get_variance()}")
	print(f"medians: {ds1.get_median()}")
	print(f"mins: {ds1.get_min()}")
	print(f"maxs: {ds1.get_max()}")
	print(ds1.summary())

