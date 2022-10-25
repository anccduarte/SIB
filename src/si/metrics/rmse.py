
import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	It computes and returns the Root-Mean-Square Error (RMSE) of the model on a
	given dataset. RMSE: sqrt((SUM[(ti - pi)^2]) / N)

	Parameters
	----------
	y_true: np.ndarray
		The true values of the labels
	y_pred: np.ndarray
		The labels predicted by a classifier
	"""
	N = y_true.shape[0]
	return np.sqrt(np.sum((y_true - y_pred) ** 2) / N)


if __name__ == "__main__":
	true = np.array([0,1,1,1,0,1])
	pred = np.array([1,0,1,1,0,1])
	print(f"RMSE: {rmse(true, pred):.4f}")

