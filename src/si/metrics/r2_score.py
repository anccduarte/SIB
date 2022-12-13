
import numpy as np

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	Computes and returns the R2 score of the model on a given dataset.
	r2_score = 1 - (SUM[(y_true - y_pred) ** 2] / SUM[(y_true - mean(y_true)) ** 2])

	Parameters
    ----------
    y_true: np.ndarray
        The true values of the target variable
    y_pred: np.ndarray
        The predicted target values
	"""
	num = np.sum((y_true - y_pred) ** 2)
	mu = np.mean(y_true)
	denom = np.sum((y_true - mu) ** 2)
	return 1 - (num / denom)


if __name__ == "__main__":
    
    true = np.array([1,2,3,4,5,6])
    pred = np.array([1,2.1,3,4.2,5,6.1])
    print(f"r2_score: {r2_score(true, pred):.2%}")

