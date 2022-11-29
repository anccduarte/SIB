
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the Mean-Squared Error (MSE) of the model on the given
    dataset. MSE: SUM[(ti - pi)^2] / 2N

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return np.sum((y_true - y_pred) ** 2) / (2*N)

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the value of the partial derivative of the Mean-Squared
    Error (MSE) with respect to y_pred. d(MSE)/d(pi): -SUM[(ti - pi)] / N.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return -np.sum(y_true - y_pred) / N


if __name__ == "__main__":

    true = np.array([0,1,1,1,0,1])
    pred = np.array([1,0,1,1,0,1])
    # MSE
    print(f"MSE: {mse(true, pred):.4f}")
    # MSE derivative with respect to y_pred
    print(f"d(MSE)/d(y_pred): {mse_derivative(true, pred):.4f}")

