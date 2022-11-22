
import numpy as np
import sys
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
                 random_state: int = None,
                 activation_function : Callable = sigmoid):
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
        random_state: int (default=None)
            Controls the random initialization of the weights matrix and/or bias vector
        activation_function: callable (default=sigmoid)
            The activation function to be used

        Attributes
        ----------
        weights: np.ndarray
            The weights matrix used in training
        bias: np.ndarray
            The bias vector used in training
        """
        # check values (sizes)
        if input_size < 1:
            raise ValueError("The value of 'input_size' must be a positive integer.")
        if output_size < 1:
            raise ValueError("The value of 'output_size' must be a positive integer.")
        # check values (initializers)
        if weights_init not in ["random", "zeros", "ones"]:
            raise ValueError("The value of 'weights_init' must be in {'random', 'zeros', 'ones'}")
        if bias_init not in ["random", "zeros", "ones"]:
            raise ValueError("The value of 'bias_init' must be in {'random', 'zeros', 'ones'}")
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.random_state = random_state
        self.activation_function = activation_function
        # attributes
        # if, initially, weights are set randomly and bias is initialized as a vector of zeros:
        # self.weights = np.random.randn(input_size, output_size) * 0.01
        # self.bias = np.zeros((1,output_size))
        self.weights, self.bias = self._init_weigths_and_bias(weights_init, bias_init)

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
        # compute the output of the layer
        z = np.dot(input_data, self.weights) + self.bias
        # compute and return the activation values of the output
        return self.activation_function(z)


if __name__ == "__main__":

    from activation import softmax

    l1 = Dense(2, 1)
    out1 = l1.forward(np.array([[1,3], [2,4]]))
    print(out1)

    l2 = Dense(2, 1, weights_init="zeros", bias_init="random")
    out2 = l2.forward(np.array([[1,3], [2,4]]))
    print(out2)

    l3 = Dense(2, 1, activation_function=softmax)
    out3 = l3.forward(np.array([[1,3], [2,4]]))
    print(out3)

