
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset
from typing import Tuple

def train_test_split(dataset: Dataset, test_size: float, random_state: int) -> Tuple[Dataset, Dataset]:
	"""
	Randomly divides a given dataset into train and test subsets. Returns new Dataset objects (a training
	set and a testing set).

	Parameters
	----------
	dataset: Dataset
		The Dataset object to be divided into train and test
	test_size: float
		The proportion of the dataset to be used for testing
	random_state: int
		Seed for the permutation generator used in the split
	"""
	n = dataset.shape()[0]
	# compute permutations
	perms = np.random.RandomState(seed=random_state).permutation(n)
	# determine the indices of the training and testing data
	n_test = int(test_size * n)
	train_idx = perms[n_test:]
	test_idx = perms[:n_test]
	# generate new datasets (one for training, another for testing)
	train = Dataset(dataset.X[train_idx], dataset.y[train_idx], dataset.features, dataset.label)
	test = Dataset(dataset.X[test_idx], dataset.y[test_idx], dataset.features, dataset.label)
	return train, test


if __name__ == "__main__":
	ds = Dataset.from_random(n_examples=10, n_features=10, label=True, seed=2)
	trn, tst = train_test_split(dataset=ds, test_size=0.3, random_state=0)
	print(trn.X.shape, trn.y.shape, tst.X.shape, tst.y.shape)

