
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset
from typing import Union

class VarianceThreshold:

    """
    Selects features according to a given variance threshold.
    Variance = SUM[(xi - xm)^2] / (n - 1) -> applied to each feature
    """

    def __init__(self, threshold: Union[int, float] = 0):
        """
        Selects features according to a given variance threshold.
        Variance = SUM[(xi - xm)^2] / (n - 1) -> applied to each feature

        Parameters
        ----------
        threshold: int, float (default=0)
            The variance threshold (the default is to only remove features which have the same
            value in all samples)

        Attributes
        ----------
        fitted: bool
            Whether the selector is already fitted
        variance: np.ndarray
            Array containing the variance of each feature of the dataset
        """
        # parameters
        if threshold < 0:
            raise ValueError("The value of 'threshold' must be greater than or equal to 0.")
        self.threshold = threshold
        # attributes
        self.fitted = False
        self.variance = None

    def fit(self, dataset: Dataset) -> "VarianceThreshold":
        """
        Fits VarianceThreshold by computing the variances of the dataset's features.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.variance = dataset.get_variance()
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by selecting the features according to the variance threshold.
        Returns a new Dataset object only containing the selected features.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        if not self.fitted:
            raise Warning("Fit 'VarianceThreshold' before calling 'transform'.")
        else:
            mask = self.variance > self.threshold
            new_X = dataset.X[:,mask]
            new_feats = np.array(dataset.features)[mask]
            return Dataset(new_X, dataset.y, list(new_feats), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits VarianceThreshold and transforms the dataset by selecting the features according to
        the variance threshold. Returns a new Dataset object only containing the selected features.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    
    # only column 1 has variance greater than 3
    X = np.array([[1,4,3],[7,5,6],[9,8,2]])
    y = np.array([10,11,12])
    ds = Dataset(X,y)
    selector = VarianceThreshold(threshold=3)
    new_ds = selector.fit_transform(ds)
    print(new_ds.X)

