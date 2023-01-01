
import numpy as np
import sys
sys.path.append("../statistics")
from sigmoid_function import sigmoid_function


# -- ACTIVATION FUNCTIONS

def identity(X: np.ndarray) -> np.ndarray:
    """
    Implements the identity activation function (the input array remains
    unchanged: f(X) = X).

    Parameters
    ----------
    X: np.ndarray
        The array of values to be activated
    """
    return X

def relu(X: np.ndarray) -> np.ndarray:
    """
    Implements the ReLU (rectified linear unit) activation function. It is
    defined as f(X) = max(0, X).

    Parameters
    ----------
    X: np.ndarray
        The array of values to be activated
    """
    return np.maximum(0, X)

def sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Implements the sigmoid activation function. It is defined as follows:
    f(X) = 1 / (1 + e^(-X)). Note: alias for sigmoid_function (statistics).

    Parameters
    ----------
    X: np.ndarray
        The array of values to be activated
    """
    return sigmoid_function(X)

def softmax(X: np.ndarray) -> np.ndarray:
    """
    Implements the softmax activation function. Converts a vector of k
    numbers into a probability distribution of k possible outcomes. It is
    defined as f(X) = e^(X - max(X)) / SUM[e^(X - max(X))].

    Parameters
    ----------
    X: np.ndarray
        The vector of values to be activated
    """
    z_exp = np.exp(X - np.amax(X))
    z_sum = np.sum(z_exp, axis=1, keepdims=True)
    return z_exp / z_sum

def tanh(X: np.ndarray) -> np.ndarray:
    """
    Implements the TanH (hyperbolic tangent) activation function. It is
    defined as f(X) = (e^X - e^(-X)) / (e^X + e^(-X)).

    Parameters
    ----------
    X: np.ndarray
        The array of values to be activated
    """
    num = np.exp(X) - np.exp(-X)
    denom = np.exp(X) + np.exp(-X)
    return num / denom


# -- ACTIVATION DERIVATIVES

def d_identity(X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the identity activation function.

    Parameters
    ----------
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    return np.ones(X.size).reshape(X.shape)

def d_relu(X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the ReLU (rectified linear unit) activation function.

    Parameters
    ----------
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    # heaviside(x1, x2) = 0 if x1 < 0 else x2 if x1 == 0 else 1 (if x1 > 0)
    return np.heaviside(X, 1)

def d_sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the sigmoid activation function.

    Parameters
    ----------
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    return sigmoid(X) * (1 - sigmoid(X))

def d_softmax(X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the softmax activation function. It is assumed that
    categorical cross-entropy is used as loss function. Combining the cross-entropy
    and softmax gradients originates the formula (y_pred - y_true), which is
    implemented in "../metrics/cross.entropy.py". Therefore, 'd_softmax' returns the
    array it takes as input.

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
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    return X

def d_tanh(X: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the TanH (hyperbolic tangent) activation function.

    Parameters
    ----------
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    return 1 - (tanh(X) ** 2)


if __name__ == "__main__":

    X = np.array([[5,10,0,-4,-6,1,2,6,8]])

    print("ACTIVATION FUNCTIONS")
    for a_func in [identity, sigmoid, tanh, softmax, relu]:
        print(a_func.__name__)
        print(a_func(X))

    print("\nACTIVATION DERIVATIVES")
    for a_func in [d_identity, d_sigmoid, d_tanh, d_softmax, d_relu]:
        print(a_func.__name__)
        print(a_func(X))

