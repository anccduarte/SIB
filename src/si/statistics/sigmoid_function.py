
import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
	"""
	Computes and returns the sigmoid function of the given input.

	Parameters
	----------
	X: np.ndarray
		The input of the sigmoid function
	"""
	return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
	X = np.array([5,10,0,-4,-6,1,2,6,8])
	X_sig = sigmoid_function(X)
	print(X_sig)

