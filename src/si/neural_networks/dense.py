
import numpy as np
from activation import sigmoid
from typing import Callable

class Dense:
    
    """
    Implements a densely-connected neural network layer. Initially, weights and bias are set
    as chosen by the user (choices are restricted to "random", "zeros" or "ones").
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 weights_init: str = "random",
                 bias_init: str = "zeros",
                 activation_function : Callable = sigmoid,
                 dropout: float = 0.0,
                 random_state: int = None):
        """
        Initializes an instance of Dense. It implements a densely-connected neural network layer.
        Initially, weights and bias are set as chosen by the user (choices are restricted to
        "random", "zeros" or "ones").

        Parameters
        ----------
        input_size: int
            The number of nodes of the input
        output_size: int
            The number of nodes of the output
        weights_init: str (default="random")
            The initializer for the weights matrix
        bias_init: str (default="zeros")
            The initializer for the bias vector
        activation_function: callable (default=sigmoid)
            The activation function to be used
        dropout: float (default=0.0)
            The percentage of neurons turned off in the layer at each step of trainig
        random_state: int (default=None)
            Controls the random initialization of the weights matrix and/or bias vector

        Attributes
        ----------
        X: np.ndarray
            The input data to forward propagate at each epoch
        weights: np.ndarray
            The weights matrix used in training
        bias: np.ndarray
            The bias vector used in training
        num_drop: int
            The number of neurons turned off in the layer at each step of training
        """
        # check values of parameters
        self._check_init(input_size, output_size, weights_init, bias_init, dropout)
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.random_state = random_state
        # attributes
        self.X = None
        self.weights, self.bias = self._init_weigths_and_bias(weights_init, bias_init)
        self.num_drop = int(dropout * self.output_size)

    @staticmethod
    def _check_init(input_size, output_size, weights_init, bias_init, dropout):
        """
        Checks values of parameters of type numeric and 'str' when initializing an instance.

        Parameters
        ----------
        input_size: int
            The number of nodes of the input
        output_size: int
            The number of nodes of the output
        weights_init: str
            The initializer for the weights matrix
        bias_init: str
            The initializer for the bias vector
        dropout: float
            The percentage of neurons turned off in the layer at each step of trainig
        """
        # check values (numeric)
        if input_size < 1:
            raise ValueError("The value of 'input_size' must be a positive integer.")
        if output_size < 1:
            raise ValueError("The value of 'output_size' must be a positive integer.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("The value of 'dropout' must be in [0,1).")
        # check values (initializers)
        poss = ["random", "zeros", "ones"]
        if weights_init not in poss:
            raise ValueError(f"The value of 'weights_init' must be in {{{', '.join(poss)}}}.")
        if bias_init not in poss:
            raise ValueError(f"The value of 'bias_init' must be in {{{', '.join(poss)}}}.")

    def _init_weigths_and_bias(self, w_init: str, b_init: str) -> tuple:
        """
        Initializes the weights matrix and the bias vector as chosen by the user. The possible
        ways of initializing both arrays are restricted to "random", "zeros" or "ones". Returns
        the initialized weights matrix and bias vector.

        Parameters
        ----------
        w_init: str
            The initializer for the weigths matrix
        b_init: str
            The initializer for the bias vector
        """
        # alias self.input_size and self.output_size
        i, o = self.input_size, self.output_size
        # enumerate initialization possibilities
        poss_init = {"random": np.random.RandomState(seed=self.random_state).randn,
                     "zeros": np.zeros,
                     "ones": np.ones}
        # initialize weigths
        func_weights = poss_init[w_init]
        weights = func_weights(i*o).reshape(i, o) * (0.01 + 0.99*(w_init!="random"))
        # initialize bias
        func_bias = poss_init[b_init]
        bias = func_bias(o)
        # return initialized weigths and bias
        return weights, bias

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes and returns the activation values of the layer's output. It performs the
        operation activation((input @ weights) + bias).

        Parameters
        ----------
        input_data: np.ndarray
            The layer's input data
        """
        # update self.X so that it can be used in 'backward'
        self.X = input_data
        # compute the output of the layer
        z = np.dot(self.X, self.weights) + self.bias
        # compute the activation values of the output
        a = self.activation_function(z)
        # dropout (return activated values with dropout)
        idx = np.random.permutation(self.output_size)[:self.num_drop]
        a[:,idx] = np.zeros((self.X.shape[0], 1))
        return a

    def backward(self, error: float, alpha: float) -> np.ndarray:
        """
        ...

        Parameters
        ----------
        error: float
            ...
        alpha: float
            ...
        """
        # self.weights -= alpha * np.dot(self.X.T, error)
        # temporary (just so that NN works)
        return error


if __name__ == "__main__":

    import sys
    sys.path.append("../io")
    from activation import relu, softmax
    from csv_file import read_csv_file

    # modified param_init and activation
    print("EX1")
    l2 = Dense(2, 1, weights_init="zeros", bias_init="random", activation_function=softmax)
    out2 = l2.forward(np.array([[1,3], [2,4]]))
    print(out2)

    # cpu (with dropout)
    print("\nEX2 - cpu")
    path = "../../../datasets/cpu/cpu.csv"
    cpu = read_csv_file(path, sep=",", features=True, label=True)
    layer_cpu = Dense(6, 4, activation_function=relu, dropout=0.5)
    out_cpu = layer_cpu.forward(cpu.X)
    print(out_cpu)

