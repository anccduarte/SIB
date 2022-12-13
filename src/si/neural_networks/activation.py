
import numpy as np


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
    f(X) = 1 / (1 + e^(-X)).

    Parameters
    ----------
    X: np.ndarray
        The array of values to be activated
    """
    return 1 / (1 + np.exp(-X))

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
    z_sum = np.sum(z_exp)
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
    Computes the derivative of the softmax activation function.

    Parameters
    ----------
    X: np.ndarray
        The array of values to which the derivative is applied
    """
    # TODO
    return np.ones(X.size).reshape(X.shape)

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

    X = np.array([5,10,0,-4,-6,1,2,6,8])

    for a_func in [identity, sigmoid, tanh, softmax, relu]:
        print(a_func.__name__)
        print(a_func(X))

