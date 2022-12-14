
import numpy as np


# -- BINARY

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the value of the binary Cross-Entropy Loss of a given model.
    Binary Cross-Entropy: -SUM[yt * log(yp) + (1 - yt) * log(1 - yp)] / N.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return -np.sum((y_true * np.log(y_pred)) + (1 - y_true) + np.log(1 - y_pred)) / N

def d_binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes and returns the value of the partial derivative of the binary Cross-Entropy
    Loss with respect to y_pred.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = y_true.shape[0]
    return ((-y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / N


# -- CATEGORICAL

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the value of the categorical Cross-Entropy Loss of a given
    model. Categorical Cross-Entropy: -SUM[ti - ln(pi)].

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return -np.sum(y_true * np.log(y_pred))

def d_categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes and returns the value of the partial derivative of the categorical
    Cross-Entropy Loss with respect to y_pred.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return y_pred - y_true


if __name__ == "__main__":

    true = np.array([0,1,1,1,0,1])
    pred = np.array([0.9,0.1,0.1,0.9,0.1,0.9])
    # Cross-Entropy
    print(f"Cross-Entropy: {binary_cross_entropy(true, pred):.4f}")
    # Cross-Entropy derivative with respect to y_pred
    print(f"d(Cross-Entropy)/d(y_pred): {d_binary_cross_entropy(true, pred)}")

