
import numpy as np
from activation import identity, relu, sigmoid, softmax, tanh
from activation import d_identity, d_relu, d_sigmoid, d_softmax, d_tanh
from typing import Callable

# activation functions and derivatives
ACTIVATION = {"identity": (identity, d_identity),
              "relu": (relu, d_relu),
              "sigmoid": (sigmoid, d_sigmoid),
              "softmax": (softmax, d_softmax),
              "tanh": (tanh, d_tanh)}

class Dense:
    
    """
    Implements a densely-connected neural network layer. Initially, weights and bias are set
    as chosen by the user (choices are restricted to "random", "zeros" or "ones"). An activation
    function is applied to the output of the layer (by default, "identity"). A dropout function
    can be used to prevent overfitting (by default, dropout=0.0).
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 weights_init: str = "random",
                 bias_init: str = "zeros",
                 activation: str = "identity",
                 dropout: float = 0.0,
                 random_state: int = None):
        """
        Initializes an instance of Dense. It implements a densely-connected neural network layer.
        Initially, weights and bias are set as chosen by the user (choices are restricted to
        "random", "zeros" or "ones"). An activation function is applied to the output of the layer
        (by default, "identity"). A dropout function can be used to prevent overfitting (by default,
        dropout=0.0).

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
        activation: str (default="identity")
            The name of the activation function to be used
        dropout: float (default=0.0)
            The percentage of neurons turned off in the layer at each step of trainig
        random_state: int (default=None)
            Controls the random initialization of the weights matrix and/or bias vector

        Attributes
        ----------
        dense_input: np.ndarray
            The input of the "dense layer"
        activation_input: np.ndarray
            The input of the "activation layer"
        weights: np.ndarray
            The weights matrix used in training
        bias: np.ndarray
            The bias vector used in training
        activation_function: callable
            The activation function to be used
        activation_derivative: callable
            The derivative of the activation function
        num_drop: int
            The number of neurons to be turned off at each step of training
        """
        # check values of parameters
        self._check_init(input_size, output_size, weights_init, bias_init, activation, dropout)
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.random_state = random_state
        # attributes
        self.dense_input = None
        self.activation_input = None
        self.weights, self.bias = self._init_weigths_and_bias(weights_init, bias_init)
        self.activation_function, self.activation_derivative = ACTIVATION[activation]
        self.num_drop = int(dropout * self.output_size)

    @staticmethod
    def _check_init(input_size, output_size, weights_init, bias_init, activation, dropout):
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
        activation: str
            The name of the activation function to be used
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
        poss_init = ["random", "zeros", "ones"]
        if weights_init not in poss_init:
            raise ValueError(f"The value of 'weights_init' must be in {{{', '.join(poss_init)}}}.")
        if bias_init not in poss_init:
            raise ValueError(f"The value of 'bias_init' must be in {{{', '.join(poss_init)}}}.")
        # check values (activation)
        poss_activation = ["identity", "relu", "sigmoid", "softmax", "tanh"]
        if activation not in poss_activation:
            raise ValueError(f"The value of 'activation' must be in {{{', '.join(poss_activation)}}}")

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
        # compute the output of the layer (dense)
        z = np.dot(input_data, self.weights) + self.bias
        # compute the activation values of the output
        a = self.activation_function(z)
        # dropout (return activated values with dropout)
        idx = np.random.permutation(self.output_size)[:self.num_drop]
        a[:,idx] = np.zeros((input_data.shape[0], 1))
        # add instance attributes so that they can be used in 'backward'
        self.dense_input = input_data
        self.activation_input = z
        # return the "activated" output
        return a

    def backward(self, error: np.ndarray, alpha: float) -> np.ndarray:
        """
        Backpropagates the error through the activation and dense "layers". Computes the "dense
        layer" error (error * f'(x)), updates the weights matrix and bias vector, and computes the
        error to propagate to the previous layer (error @ w.T). Returns the error to propagate.

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        # compute error to propagate to dense
        if self.activation == "softmax": error_prop_dense = error
        else: error_prop_dense = error * self.activation_derivative(self.activation_input)
        # update weights and bias of dense
        self.weights -= alpha * np.dot(self.dense_input.T, error_prop_dense)
        self.bias -= alpha * np.sum(error_prop_dense, axis=0)
        # compute and return the error to propagate to the next layer
        error_prop_next = np.dot(error_prop_dense, self.weights.T)
        return error_prop_next


if __name__ == "__main__":

    import sys
    sys.path.append("../io")
    from csv_file import read_csv_file

    # modified param_init and activation
    print("EX1")
    l = Dense(2, 1, weights_init="zeros", bias_init="random", activation="softmax")
    out = l.forward(np.array([[1,3], [2,4]]))
    print(out)

    # cpu (with dropout)
    print("\nEX2 - cpu")
    path = "../../../datasets/cpu/cpu.csv"
    cpu = read_csv_file(path, sep=",", features=True, label=True)
    layer_cpu = Dense(6, 4, activation="relu", dropout=0.5)
    out_cpu = layer_cpu.forward(cpu.X)
    print(out_cpu)

