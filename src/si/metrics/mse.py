
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

    Returns
    -------
    mse: float
        The mean squared error of the model
    """
    N = y_true.shape[0]
    return np.sum((y_true - y_pred) ** 2) / (2*N)


if __name__ == "__main__":
    true = np.array([0,1,1,1,0,1])
    pred = np.array([1,0,1,1,0,1])
    print(f"MSE: {mse(true, pred):.4f}")

