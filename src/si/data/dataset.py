
import numpy as np
import pandas as pd
from typing import Tuple

class Dataset:

    """
    Constructs a tabular dataset for machine learning. The dataset is built from a set of feature
    vectors (X) and an optional label vector (y).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: list = None, label: str = None):
        """
        Constructs a tabular dataset for machine learning. The dataset is built from a set of feature
        vectors (X) and an optional label vector (y).

        Parameters
        ----------
        X: np.ndarray
            The matrix containing the dataset's feature vector(s)
        y: np.ndarray (default=None)
            The label vector
        features: list (deafult=None)
            The name(s) of the feature(s)
        label: str (default=None)
            The name of the label
        """
        # check dimensions
        self._check_init(X, y, features)
        # parameters
        # reshape X if it only has one feature (otherwise, it defaults to a one dimensional array)
        self.X = X if X.ndim > 1 else np.reshape(X, (-1, 1))
        self.y = y
        self.features = [f"feat{i+1}" for i in range(X.shape[1])] if features is None else features
        self.label = "label" if (y is not None and label is None) else label

    @staticmethod
    def _check_init(X: np.ndarray, y: np.ndarray, features: list):
        """
        Checks whether the dimensions of the arrays it takes as parameters are correct.

        Parameters
        ----------
        X: np.ndarray
            The matrix containing the dataset's feature vector(s)
        y: np.ndarray
            The label vector
        features: list
            The name(s) of the feature(s)
        """
        if y is not None:
            if X.shape[0] != y.size:
                raise ValueError("The number of examples in 'X' must be equal to the size of 'y'.")
        if features:
            if X.shape[1] != len(features):
                raise ValueError("The number of features in 'X' must be equal to len(features).")

    def shape(self) -> Tuple[int, int]:
        """
        Returns a two-element tuple consisting of the dataset's dimensions.
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns a boolean value representative of the presence of a label.
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the unique classes of the dataset.
        """
        if self.y is None:
            raise ValueError("The parameter 'y' was set to 'None'.")
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the mean of each feature.
        """
        return self.X.mean(axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the variance of each feature.
        """
        return self.X.var(axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the median of each feature.
        """
        return np.median(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the minimum value of each feature.
        """
        return self.X.min(axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns an np.ndarray containing the maximum value of each feature.
        """
        return self.X.max(axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a pd.DataFrame containing some descriptive metrics (mean, variance, median,
        minimum value and maximum value) of each feature.
        """
        df = pd.DataFrame({"Mean": self.get_mean(),
                           "Variance": self.get_variance(),
                           "Median": self.get_median(),
                           "Min": self.get_min(),
                           "Max": self.get_max()})
        return df

    @classmethod
    def from_random(cls,
                    n_examples: int,
                    n_features: int,
                    label: bool,
                    features_range: tuple = (1, 10),
                    label_range: tuple = (0, 3),
                    seed: int = None) -> 'Dataset':
        """
        Randomly generates and returns a new Dataset object based on given parameters. By default,
        the values of the dataset's features range from 1 to 9. If present, the values of the
        dataset's label range, by deafult, from 0 to 2.

        Parameters
        ----------
        n_examples: int
            Number of examples in the dataset
        n_features: int
            Number of features in the dataset
        label: bool
            Whether the dataset has a label
        feature_range: tuple (default=(1, 10))
            The interval of values used when generating the feature vectors
        label_range: tuple (default=(0, 3))
            The interval of values used when generating the label vector (ignored when label := None)
        seed: int (default=None)
            Seed for the random number generator
        """
        lf, uf = features_range
        ll, ul = label_range
        X = np.random.RandomState(seed=seed).randint(lf, uf, (n_examples, n_features))
        y = np.random.RandomState(seed=seed).randint(ll, ul, (n_examples,)) if label else None
        return cls(X, y)

    def remove_nan(self):
        """
        Removes examples which contain missing values (NaN).
        """
        idx = np.isnan(self.X).any(axis=1)
        self.X = self.X[~idx]
        if self.y is not None:
            self.y = self.y[~idx]

    def fill_nan(self, fill: str):
        """
        Replaces all dataset's missing values (NaN) by the mean/median of the respective column.
        Allowed values for 'fill' are:
            - 'mean': calls np.nanmean on the matrix containing the features
            - 'median': calls np.nanmedian on the matrix containig the features

        Parameters
        ----------
        fill: str
            The string description of the value by which missing values are replaced
        """
        vals = {"mean": np.nanmean, "median": np.nanmedian}
        fill_func = vals[fill]
        self.X = np.nan_to_num(self.X, nan=fill_func(self.X, axis=0))


if __name__ == "__main__":

    print("EX1")
    X1 = np.array([[1,2,3],[4,5,6]])
    y1 = np.array([1,2])
    f1 = ["A","B","C"]
    l1 = "y"
    ds1 = Dataset(X1, y1, f1, l1)
    print(f"shape: {ds1.shape()}")
    print(f"has_label: {ds1.has_label()}")
    print(f"classes: {ds1.get_classes()}")
    print(ds1.summary())

    print("\nEX2")
    X2 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
    y2 = np.array([7,8,9])
    ds2 = Dataset(X2, y2)
    print(f"shape (before removing NaNs): {ds2.shape()}")
    ds2.remove_nan()
    print(f"shape (after removing NaNs): {ds2.shape()}")
    print(ds2.summary())

    print("\nEX3")
    X3 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
    y3 = np.array([7,8,9])
    ds3 = Dataset(X3, y3)
    print(f"shape (before filling NaNs): {ds3.shape()}")
    ds3.fill_nan("mean")
    print(f"shape (after filling NaNs): {ds3.shape()}")
    print(ds3.summary())

    print("\nEX4")
    ds4 = Dataset.from_random(n_examples=10, n_features=4, label=True, seed=0)
    print(f"shape: {ds4.shape()}")
    print(f"has_label: {ds4.has_label()}")
    print(f"classes: {ds4.get_classes()}")
    print(ds4.summary())

    print("\nEX5")
    ds5 = Dataset.from_random(n_examples=20,
                              n_features=6,
                              label=True,
                              features_range=(10, 20),
                              label_range=(0, 2),
                              seed=0)
    print(f"shape: {ds5.shape()}")
    print(f"has_label: {ds5.has_label()}")
    print(f"classes: {ds5.get_classes()}")
    print(ds5.summary())

