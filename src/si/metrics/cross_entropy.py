
import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the value of the Cross-Entropy Loss of a given model.
    Cross-Entropy: -SUM[ti - ln(pi)] / N.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / N

def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the value of the partial derivative of the Cross-Entropy
    Loss with respect to y_pred. d(Cross-Entropy)/d(pi): SUM[1/pi] / N.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return np.sum(1 / y_pred) / N


if __name__ == "__main__":

	# ERROR -> division by 0
    true = np.array([0,1,1,1,0,1])
    pred = np.array([1,0,1,1,0,1])
    # Cross-Entropy
    print(f"Cross-Entropy: {cross_entropy(true, pred):.4f}")
    # Cross-Entropy derivative with respect to y_pred
    print(f"d(Cross-Entropy)/d(y_pred): {cross_entropy_derivative(true, pred):.4f}")

