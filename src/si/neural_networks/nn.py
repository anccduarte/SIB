
import numpy as np
import sys
PATHS = ["../data", "../metrics"]
sys.path.extend(PATHS)
import warnings
from cross_entropy import binary_cross_entropy, d_binary_cross_entropy
from cross_entropy import categorical_cross_entropy, d_categorical_cross_entropy
from dataset import Dataset
from mse import mse, mse_derivative
from typing import Callable, Union

# loss functions and the respective derivatives
LOSS = {"mse": (mse, mse_derivative),
        "binary_cross_entropy": (binary_cross_entropy, d_binary_cross_entropy),
        "categorical_cross_entropy": (categorical_cross_entropy, d_categorical_cross_entropy)}

class NN:

    """
    Implements a multi-layered neural network model which trains using backpropagation. 
    """

    def __init__(self,
                 layers: Union[tuple, list],
                 epochs: int = 1000,
                 num_batches: int = 10,
                 alpha: float = 0.001,
                 loss: str = "mse",
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
        loss: str (default="mse")
            The name of the loss function to be used
        random_state: int (default=None)
            Controls the shuffling of the dataset before the split into batches
        verbose: bool (default=False)
            Whether to print the value of the loss function at each epoch

        Attributes
        ----------
        fitted: bool
            Wheter 'NN' is already fitted
        loss_function: callable
            The loss function used to backpropagate the error
        loss_derivative: callable
            The derivative of the loss function
        history: dict
            A dictionary object storing the values of the loss function at each epoch
        """
        # check numeric parameters
        self._check_init(epochs, num_batches, alpha, loss)
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.num_batches = num_batches
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        # attributes
        self.fitted = False
        self.loss_function, self.loss_derivative = LOSS[loss]
        self.history = {}

    @staticmethod
    def _check_init(epochs, num_batches, alpha, loss):
        """
        Checks values of numeric and 'str' parameters when initializing an instance.

        Parameters
        ----------
        epochs: int
            The maximum number of training epochs
        num_batches: int
            The number of batches to use when spliting the training data
        alpha: float
            The learning-rate used in training
        loss: str
            The name of the loss function to be used
        """
        # check values (numeric)
        if epochs < 1:
            raise ValueError("The value of 'epochs' must be a positive integer.")
        if num_batches < 1:
            raise ValueError("The value of 'num_batches' must be a positive integer.")
        if alpha <= 0 or alpha > 1:
            raise ValueError("The value of 'alpha' must be in (0,1].")
        # check values (str)
        loss_poss = ["mse", "binary_cross_entropy", "categorical_cross_entropy"]
        if loss not in loss_poss:
            raise ValueError(f"The value of 'loss' must be in {{{', '.join(loss_poss)}}}.")

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
        # get permutations, shuffle the dataset and reshape dataset.y
        shuffle = np.random.RandomState(seed=self.random_state).permutation(num_examples)
        x = dataset.X[shuffle]
        y = dataset.y.reshape(-1, 1)[shuffle]
        # get batches
        size_batch = num_examples // self.num_batches
        x_batches = [x[i:i+size_batch] for i in range(0, num_examples, size_batch)]
        y_batches = [y[i:i+size_batch] for i in range(0, num_examples, size_batch)]
        # fix last batch (if the batch size is too small, concatenate to second-last batch)
        if 0 < num_examples % self.num_batches < size_batch // 2:
            x_batches[-1] = np.concatenate((x_batches[-2], x_batches.pop(-1)), axis=0)
            y_batches[-1] = np.concatenate((y_batches[-2], y_batches.pop(-1)), axis=0)
        return x_batches, y_batches

    def fit(self, dataset: Dataset) -> "NN":
        """
        Fits the model to the dataset using forward and backward propagation along its layers.
        This process is repeated <self.epochs> times. Returns self.

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
            for batch, (x, y) in enumerate(zip(x_batches, y_batches), start=1):
                # copy x so that x_batches is not altered
                # y_pred (last layer -> predictions)
                y_pred = x.copy()
                # forward -> the output of one layer is the input of its successor
                for layer in self.layers:
                    y_pred = layer.forward(y_pred)
                # compute error
                error = self.loss_derivative(y, y_pred)
                # backpropagate error
                for layer in self.layers[::-1]:
                    error = layer.backward(error, self.alpha)
                # save batch loss in history
                loss = self.loss_function(y, y_pred)
                self.history[epoch][batch] = loss
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
        y_pred = dataset.X.copy()
        for layer in self.layers:
            y_pred = layer.forward(y_pred)
        # reshape y_pred to an array of one row (=dataset.y)
        return y_pred.reshape(dataset.y.shape)

    def score(self, dataset: Dataset, score_func: Callable) -> float:
        """
        Computes and returns the score of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the score on)
        score_func: callable
            The scoring function to be used
        """
        y_pred = self.predict(dataset)
        return score_func(dataset.y, y_pred)


if __name__ == "__main__":
    ...

