
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
    return -np.sum((y_true * np.log(y_pred)) + (1 - y_true) * np.log(1 - y_pred)) / N

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
    model. Categorical Cross-Entropy: -SUM[ti * ln(pi)].

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
    Cross-Entropy Loss with respect to y_pred. It is assumed that softmax activation is
    used in the last layer of the neural network. Combining the cross-entropy and
    softmax gradients originates the formula (y_pred - y_true), which is the function's
    return value. Therefore, 'd_softmax', the derivative of the softmax activation
    function (implemented in "../neural_networks/activation.py"), returns the array it
    takes as input (X).

    Formulas:
    - softmax -> y_pred_i = exp(x_i) / SUM(k)[exp(x_k)]
    - categorical cross-entropy -> L = -SUM(i)[y_true_i * ln(y_pred_i)]

    Derivative of softmax:
    - i = j
      d(y_pred_i)/d(x_i) = 
      = (exp(x_i) * SUM(k)[exp(x_k)] - exp(x_i)^2) / (SUM(k)[exp(x_k)])^2 =
      = y_pred_i * (1 - y_pred_i)
    - i != j
      d(y_pred_i)/d(x_j) =
      = (0 - exp(x_i) * exp(x_j)) / (SUM(k)[exp(x_k)])^2 =
      = - y_pred_i * y_pred_j

    Derivative of categorical cross-entropy:
    - d(L)/d(y_pred_i) = -SUM(i)[y_true_i - (1 / y_pred_i)]

    Combining gradients:
    - d(L)/d(x_j) =
      = - (SUM(i!=j)[y_true_i - (1 / y_pred_i)] * d(y_pred_i)/d(x_j) +
        + y_true_j * (1 / y_pred_j) * d(y_pred_j)/d(x_j)) =
      = - (SUM(i!=j)[y_true_i - (1 / y_pred_i)] * (-y_pred_i * y_pred_j) +
        + y_true_j * (1 / y_pred_j) * y_pred_j * (1 - y_pred_j)) =
      = SUM(i!=j)[y_true_i * y_pred_j] + y_true_j * y_pred_j - y_true_j =
      = SUM(i)[y_true_i * y_pred_j] - y_true_j =
      = y_pred_j - y_true_j

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return (y_pred - y_true)


if __name__ == "__main__":

    print("BINARY")
    true = np.array([0,1,1,1,0,1])
    pred = np.array([0.9,0.1,0.1,0.9,0.1,0.9])
    # Cross-Entropy (binary)
    print(f"Cross-Entropy: {binary_cross_entropy(true, pred):.4f}")
    # Cross-Entropy derivative with respect to y_pred (binary)
    print(f"d(Cross-Entropy)/d(y_pred): {d_binary_cross_entropy(true, pred)}")

    print("\nCATEGORICAL")
    true_c = np.array([[1,0,0], [0,1,0], [0,0,1]])
    pred_c = np.array([[0.4,0.25,0.35], [0.2,0.6,0.2], [0.3,0.2,0.5]])
    # Cross-Entropy (categorical)
    print(f"Cross-Entropy: {categorical_cross_entropy(true_c, pred_c):.4f}")
    # Cross-Entropy derivative with respect to y_pred (categorical)
    print(f"d(Cross-Entropy)/d(y_pred): {d_categorical_cross_entropy(true_c, pred_c)}")

