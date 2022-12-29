
import numpy as np
import sys
PATHS = ["../data", "../metrics"]
sys.path.extend(PATHS)
from accuracy import accuracy
from dataset import Dataset

class VotingClassifier:

    """
    Implements an ensemble model which uses voting as the combination function. If
    applicable, each prediction vector is weighted by the score of the model which
    produced it.
    """

    def __init__(self, models: list, weighted: bool = False):
        """
        Implements an ensemble model which uses voting as the combination function. If
        applicable, each prediction vector is weighted by the score of the model which
        produced it.

        Parameters
        ----------
        models: list
            A list object containing initialized instances of classifiers
        weighted: bool (default=False)
            Whether to weigh model predictions by the respective scores

        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        """
        # parameters
        self.models = models
        self.weighted = weighted
        # attributes
        self.fitted = False

    def fit(self, dataset: Dataset) -> "VotingClassifier":
        """
        Fits VotingClassifier by fitting the models in <self.models>. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset used to fit the model)
        """
        for model in self.models:
            model.fit(dataset)
        self.fitted = True
        return self

    def _get_majority_vote(self, predictions: np.ndarray) -> int:
        """
        Helper function which determines and returns the most common label in a set
        of predictions.

        Parameters
        ----------
        predictions: np.ndarray
            An array consisting of the labels predicted for a given example
        """
        labels, counts = np.unique(predictions, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset. To do so, it uses voting to
        combine the predictions of the models in <self.models>. If <self.weighted> is
        set to True, model predictions are weighted according to the respective scores.
        Note: assumes that all models use the same scoring metric.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset containing the examples to be labeled)
        """
        if not self.fitted:
            raise Warning("Fit 'VotingClassifier' before calling 'predict'.")
        # array containing the outputs of each k models in k rows
        predictions = np.array([model.predict(dataset) for model in self.models])
        # weigh model predictions based on the respective scores
        if self.weighted:
            # get model scores
            scores = [model.score(dataset) for model in self.models]
            # scale scores so that min_score = 1
            min_scr = min(scores)
            scores_sc = np.array([round((1 / min_scr) * scr) for scr in scores])
            # update predictions in order to account for the computed weights
            predictions = np.repeat(predictions, repeats=scores_sc, axis=0)
        # voting is performed col-wise so that outputs of different models are compared
        return np.apply_along_axis(self._get_majority_vote, axis=0, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy score of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the accuracy on)
        """
        if not self.fitted:
            raise Warning("Fit 'VotingClassifier' before calling 'score'.")
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../linear_model", "../model_selection", "../neighbors"]
    sys.path.extend(TEST_PATHS)
    from csv_file import read_csv_file
    from knn_classifier import KNNClassifier
    from logistic_regression import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from split import train_test_split

    path_to_file = "../../../datasets/breast/breast-bin.csv"
    breast = read_csv_file(file=path_to_file, sep=",", features=False, label=True)
    breast.X = StandardScaler().fit_transform(breast.X)
    breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)

    models = [KNNClassifier(), LogisticRegression()]
    vc = VotingClassifier(models=models, weighted=True)
    vc = vc.fit(breast_trn)
    predictions = vc.predict(breast_tst)
    print(f"Predictions:\n{predictions}")
    score = vc.score(breast_tst)
    print(f"\nScore: {score*100:.2f}%")

