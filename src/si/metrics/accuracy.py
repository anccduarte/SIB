
import numpy as np
import sys
sys.path.append("../data")

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	It computes and returns the accuracy score of the model on a given dataset.
	Accuracy score: (TP+TN)/(TP+FP+TN+FN)

	Parameters
	----------
	y_true: np.ndarray
		The true values of the labels
	y_pred: np.ndarray
		The labels predicted by a classifier
	"""
	return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
	true = np.array([0,1,1,1,0,1])
	pred = np.array([1,0,1,1,0,1])
	print(f"Accuracy score: {accuracy(true, pred)*100:.2f}%")

