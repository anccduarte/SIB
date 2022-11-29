
import numpy as np
import sys
PATHS = ["../data", "../metrics"]
sys.path.extend(PATHS)
import warnings
from dataset import Dataset
from mse import mse, mse_derivative
from typing import Callable, Union

class NN:

    """
    Implements a multi-layered neural network model which trains using backpropagation. 
    """

    def __init__(self,
                 layers: Union[tuple, list],
                 epochs: int = 1000,
                 num_batches: int = 10,
                 alpha: float = 0.001,
                 loss_function: Callable = mse,
                 loss_derivative: Callable = mse_derivative,
                 random_state: int = None,
                 verbose: bool = False):
        """
        Initializes an instance of NN. It implements a multi-layered neural network model
        which trains using backpropagation.
        
        Parameters
        ----------
        layers: tuple, list
            The layers composing the neural network
        epochs: int (default=1000)
            The maximum number of training epochs
        num_batches: int (default=10)
            The number of batches to use when spliting the training data. Depending on the
            number of examples in the training dataset, num_batches := num_batches + 1.
        alpha: float (default=0.001)
            The learning-rate used in training
        loss_function: callable (default=mse)
            The loss function used to backpropagate the error
        loss_derivative: callable (default=mse_derivative)
            The derivative of the loss function
        random_state: int (default=None)
            Controls the shuffling of the dataset before the split into batches
        verbose: bool (default=False)
            Whether to print the value of the loss function at each epoch

        Attributes
        ----------
        fitted: bool
            Wheter 'NN' is already fitted
        history: dict
            A dictionary object storing the value of the loss function at each epoch
        """
        # check numeric parameters
        self._check_init(epochs, num_batches, alpha)
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.num_batches = num_batches
        self.alpha = alpha
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
        self.random_state = random_state
        self.verbose = verbose
        # attributes
        self.fitted = False
        self.history = {}

    @staticmethod
    def _check_init(epochs, num_batches, alpha):
        """
        Checks values of numeric type parameters when initializing an instance.

        Parameters
        ----------
        epochs: int
            The maximum number of training epochs
        num_batches: int
            The number of batches to use when spliting the training data
        alpha: float
            The learning-rate used in training
        """
        if epochs < 1:
            raise ValueError("The value of 'epochs' must be a positive integer.")
        if num_batches < 1:
            raise ValueError("The value of 'num_batches' must be a positive integer.")
        if alpha <= 0 or alpha > 1:
            raise ValueError("The value of 'alpha' must be in (0,1].")

    def _get_batches(self, dataset: Dataset) -> tuple:
        """
        Splits the training dataset into batches. Returns a tuple containing the feature
        batches (x_batches) and the label batches (y_batches).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the data to be split into batches)
        """
        num_examples = dataset.shape()[0]
        # check if num_batches is valid
        if num_examples < self.num_batches:
            w_msg = "The value of 'num_batches' cannot be greater than dim 0 of 'dataset'."
            s_msg = f" Setting 'num_batches' to {num_examples}..."
            warnings.warn(w_msg+s_msg, Warning)
            self.num_batches = num_examples
        # get permutations and shuffle the dataset
        shuffle = np.random.RandomState(seed=self.random_state).permutation(num_examples)
        x = dataset.X.copy()[shuffle]
        y = dataset.y.copy()[shuffle]
        # get batches
        size_batch = num_examples // self.num_batches
        x_batches = [x[i:i+size_batch] for i in range(0, num_examples, size_batch)]
        y_batches = [y[i:i+size_batch] for i in range(0, num_examples, size_batch)]
        return x_batches, y_batches

    def fit(self, dataset: Dataset) -> "NN":
        """
        Fits the model to the dataset using forward propagation along its layers.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be fitted)
        """
        # get data batches
        x_batches, y_batches = self._get_batches(dataset)
        # main loop -> <self.epochs> forward and backwards propagations
        for epoch in range(1, self.epochs+1):
            # initialize an empty dictionary for every epoch
            self.history[epoch] = {}
            # go through every batch of data
            for i, (x, y) in enumerate(zip(x_batches, y_batches), start=1):
                # forward -> the output of one layer is the input of its successor
                for layer in self.layers:
                    x = layer.forward(x)
                # compute error
                error = self.loss_derivative(y, x)
                # backpropagate error
                for layer in self.layers[::-1]:
                    error = layer.backward(error, self.alpha)
                # save batch loss in history
                loss = self.loss_function(y, x)
                self.history[epoch][i] = loss
            # print loss (mean of losses at epoch <epoch>)
            if self.verbose:
                mean_loss = np.mean(list(self.history[epoch].values()))
                print(f"Epoch {epoch}/{self.epochs} -- {mean_loss = :.4f}")
        self.fitted = True
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to predict the output of)
        """
        if not self.fitted:
            raise Warning("Fit 'NN' before calling 'predict'.")
        # forward propagate with learned parameters
        x = dataset.X.copy()
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":

    sys.path.append("../io")
    from activation import identity, relu
    from csv_file import read_csv_file
    from dense import Dense

    # data (cpu)
    path = "../../../datasets/cpu/cpu.csv"
    cpu = read_csv_file(path, sep=",", features=True, label=True)

    # -- split data into train and test when NN is complete --

    # layers
    # if SigmoidActivation is implemented in its own class:
    # layers = [l1, SigmoidActivation(), l2, SigmoidActivation()]
    l1 = Dense(input_size=6, output_size=4, activation_function=relu)
    l2 = Dense(input_size=4, output_size=1, activation_function=identity)
    layers = [l1, l2]

    # NN model
    nn_model = NN(layers=layers, epochs=10, num_batches=4, verbose=True)
    # -- use train and test when NN is complete --
    nn_model.fit(cpu)
    preds = nn_model.predict(cpu)
    print("\nPredictions:")
    print(preds)

