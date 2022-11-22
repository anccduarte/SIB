
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset
from typing import Union

class NN:

    """
    Implements a multi-layered neural network model which trains using backpropagation. 
    """

    def __init__(self, layers: Union[tuple, list]):
        """
        Initializes an instance of NN. It implements a multi-layered neural network model
        which trains using backpropagation.
        
        Parameters
        ----------
        layers: tuple, list
            The layers composing the neural network

        Attributes
        ----------
        fitted: bool
            Wheter 'NN' is already fitted
        """
        # parameters
        self.layers = layers
        # attributes
        self.fitted = False

    def fit(self, dataset: Dataset) -> np.ndarray:
        """
        Fits the model to the dataset using forward propagation along its layers.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be fitted)
        """
        x = dataset.X.copy()
        # the output of one layer is the input of its successor
        for layer in self.layers:
            x = layer.forward(x)
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
        ...


if __name__ == "__main__":

    from dense import Dense

    # layers
    l1 = Dense(input_size=2, output_size=2)
    l2 = Dense(input_size=2, output_size=1)
    layers = [l1, l2]
    # if SigmoidActivation is implemented in its own class:
    # layers = [l1, SigmoidActivation(), l2, SigmoidActivation()]
    
    # data
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([1,0,0,1])
    ds = Dataset(x, y)

    # NN model
    nn_model = NN(layers=layers)
    nn_model.fit(ds)
    # nn_model.predict(...)

