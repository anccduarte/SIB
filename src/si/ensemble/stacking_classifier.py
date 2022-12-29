
import numpy as np
import sys
PATHS = ["../data", "../metrics", "../neighbors"]
sys.path.extend(PATHS)
from accuracy import accuracy
from dataset import Dataset
from knn_classifier import KNNClassifier
from typing import Union

class StackingClassifier:

    """
    Implements an ensemble model, which uses a stack of models to train a final classifier.
    The stack of models is built by stacking the output (predictions) of each model. If
    applicable, each prediction vector is weighted by the score of the model which produced it.
    """

    def __init__(self, models: list, final_model = KNNClassifier, weighted: bool = False):
        """
        Implements an ensemble model, which uses a stack of models to train a final classifier.
        The stack of models is built by stacking the output (predictions) of each model. If
        applicable, each prediction vector is weighted by the score of the model which produced it.

        Parameters
        ----------
        models: list
            A list object containing initialized instances of classifiers
        final_model: classifier (default=KNNClassifier)
            The final classifier to be used
        weighted: bool (default=False)
            Whether to weigh model predictions by the respective scores

        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        scores_sc: np.ndarray
            The scaled scores of each model trained during fit
        """
        # parameters
        self.models = models
        self.final_model = final_model
        self.weighted = weighted
        # attributes
        self.fitted = False
        self.scores_sc = None

    def fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Fits StackingClassifier. To do so:
        1. Fits the models of the ensemble (self.models)
        2. Predicts labels based on those models
        3. If applicable, weighs predictions based on model scores
        4. Fits the final model based on the computed predictions

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset used to fit the model)
        """
        # fit the ensemble on training data (1)
        for model in self.models:
            model.fit(dataset)
        # get the predictions of each model for training data (2)
        predictions = np.array([model.predict(dataset) for model in self.models])
        # weigh model predictions based on the respective scores (3)
        if self.weighted:
            # get model scores
            scores = [model.score(dataset) for model in self.models]
            # scale scores so that min_score = 1 (store scores_sc -> use in 'predict')
            min_scr = min(scores)
            self.scores_sc = np.array([round((1 / min_scr) * scr) for scr in scores])
            # update predictions in order to account for the computed weights
            predictions = np.repeat(predictions, repeats=self.scores_sc, axis=0)
        # create a Dataset object containing the predictions
        ds_train = Dataset(predictions.T, dataset.y)
        # fit the final model based on the predictions of the ensemble (4)
        self.final_model.fit(ds_train)
        self.fitted = True
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset. The predictions are made according to
        <self.final_model> trained on the (weighted) predictions of <self.models>.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset containing the examples to be labeled)
        """
        if not self.fitted:
            raise Warning("Fit 'StackingClassifier' before calling 'predict'.")
        # get the predictions of each model for testing data
        predictions = np.array([model.predict(dataset) for model in self.models])
        # weigh model predictions based on the scores obtained during training
        if self.weighted:
            predictions = np.repeat(predictions, repeats=self.scores_sc, axis=0)
        # create a Dataset object containing the predictions
        ds_test = Dataset(predictions.T, dataset.y)
        # return the predictions
        return self.final_model.predict(ds_test)

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy score of the final model on the dataset.
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the accuracy on)
        """
        if not self.fitted:
            raise Warning("Fit 'StackingClassifier' before calling 'score'.")
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../linear_model", "../model_selection"]
    sys.path.extend(TEST_PATHS)
    from csv_file import read_csv_file
    from logistic_regression import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from split import train_test_split

    path_to_file = "../../../datasets/breast/breast-bin.csv"
    breast = read_csv_file(file=path_to_file, sep=",", features=False, label=True)
    breast.X = StandardScaler().fit_transform(breast.X)
    breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)

    models = [KNNClassifier(), LogisticRegression()]
    sc = StackingClassifier(models=models, final_model=KNNClassifier(), weighted=True)
    sc = sc.fit(breast_trn)
    predictions = sc.predict(breast_tst)
    print(f"Predictions: {predictions}")
    score = sc.score(breast_tst)
    print(f"\nScore: {score*100:.2f}%")

