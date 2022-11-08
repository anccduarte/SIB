
import numpy as np
import sys
PATHS = ["../data", "../statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from f_classification import f_classification
from typing import Callable

class SelectPercentile:

    """
    Selects the best features according to a given percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value of each feature with examples grouped by label
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.2):
        """
        Selects the best features according to a given percentile.
        Feature ranking is performed by computing the scores of each feature using a scoring function:
            - f_classification: ANOVA F-value of each feature with examples grouped by label

        Parameters
        ----------
        score_func: callable (default=f_classification)
            Function that takes a Dataset object and returns a tuple of arrays (F-scores and p-values)
        percentile: float (default=0.2)
            The percentage of features to be selected

        Attributes
        ----------
        fitted: bool
            Whether the selector is already fitted
        F: np.ndarray
            The F-score(s) of feature(s)
        p: np.ndarray
            The p-value(s) of F-score(s)
        """
        # parameters
        if percentile < 0 or percentile > 1:
            raise ValueError("The value of 'percentile' must be in [0,1].")
        self.score_func = score_func
        self.percentile = percentile
        # attributes
        self.fitted = False
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Fits SelectPercentile by computing the F-scores and p-values of the dataset's features.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A labeled Dataset object
        """
        self.F, self.p = self.score_func(dataset)
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by selecting the best features according to a given percentile.
        Returns a new Dataset object only containing the selected features.

        Parameters
        ----------
        dataset: Dataset
            A labeled Dataset object
        """
        if not self.fitted:
            raise Warning("Fit 'SelectPercentile' before calling 'transform'.")
        else:
            n_feats = round(len(dataset.features)*self.percentile)
            idxs = np.argsort(self.F)[-n_feats:]
            new_X = dataset.X[:,idxs]
            new_feats = np.array(dataset.features)[idxs]
            return Dataset(new_X, dataset.y, list(new_feats), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits SelectPercentile and transforms the dataset by selecting the best features according
        to a given percentile. Returns a new Dataset object only containing the selected features.

        Parameters
        ----------
        dataset: Dataset
            A labeled Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":

    X = np.array([[1,2,3,4],[3,6,5,1],[7,4,1,5],[1,3,2,9]])
    y = np.array([1,1,0,0])
    ds = Dataset(X,y)
    selector = SelectPercentile(score_func=f_classification, percentile=0.4)
    new_ds = selector.fit_transform(ds)
    print(new_ds.X)

