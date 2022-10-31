
import numpy as np
import sys
PATHS = ["../data", "../linear_model", "../metrics", "../neighbors"]
sys.path.extend(PATHS)
from accuracy import accuracy
from dataset import Dataset
from knn_classifier import KNNClassifier
from logistic_regression import LogisticRegression
from typing import Union

class StackingClassifier:

	"""
	Implements an ensemble model, which uses a stack of models to train a final classifier.
	The stack of models is built by stacking the output (predictions) of each model.
	"""

	def __init__(self, models: list, final_model: Union[KNNClassifier, LogisticRegression]):
		"""
		Implements an ensemble model, which uses a stack of models to train a final classifier.
		The stack of models is built by stacking the output (predictions) of each model.

		Parameters
		----------
		models: list
			A list object containing initialized instances of classifiers
		final_model: KNNClassifier | LogisticRegression
			The final classifier

		Attributes
		----------
		fitted: bool
			Whether the model is already fitted
		"""
		# parameters
		self.models = models
		self.final_model = final_model
		# attributes
		self.fitted = False

	def fit(self, dataset: Dataset) -> "StackingClassifier":
		"""
		Fits StackingClassifier. To do so:
		1. Fits the models of the ensemble (self.models)
		2. Predicts labels based on those models
		3. Fits the final model based on the latter predictions

		Parameters
		----------
		dataset: Dataset
			A Dataset object (the dataset used to fit the model)
		"""
		# fit the ensemble on trainig data
		for model in self.models:
			model.fit(dataset)
		# get the predictions of each model for trainig data
		predictions = np.array([model.predict(dataset) for model in self.models])
		# create a Dataset object
		ds = Dataset(predictions.T, dataset.y)
		# fit the final model based on the predictions of the ensemble
		self.final_model.fit(ds)
		self.fitted = True
		return self

	def predict(self, dataset: Dataset) -> np.ndarray:
		"""
		Predicts and returns the output of the dataset. The predictions are made according
		to self.final_model trained on the predictions of self.models.

		Parameters
		----------
		dataset: Dataset
			A Dataset object (the dataset containing the examples to be labeled)
		"""
		if not self.fitted:
			raise Warning("Fit 'StackingClassifier' before calling 'predict'.")
		# get the predictions of each model for testing data
		predictions = np.array([model.predict(dataset) for model in self.models])
		# create a Dataset object
		ds = Dataset(predictions.T, dataset.y)
		# return the predictions
		return self.final_model.predict(ds)

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

	TEST_PATHS = ["../io", "../model_selection", "../statistics"]
	sys.path.extend(TEST_PATHS)
	from csv_file import read_csv_file
	from distances import euclidean_distance
	from sklearn.preprocessing import StandardScaler
	from split import train_test_split

	path_to_file = "../../../datasets/breast/breast-bin.csv"
	breast = read_csv_file(file=path_to_file, sep=",", features=False, label=True)
	breast.X = StandardScaler().fit_transform(breast.X)
	breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)

	models = [KNNClassifier(k=4, distance=euclidean_distance), 
			  LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=0.0001, adaptative_alpha=False)]

	sc = StackingClassifier(models, KNNClassifier(k=4, distance=euclidean_distance))
	sc = sc.fit(breast_trn)
	predictions = sc.predict(breast_tst)
	print(predictions)
	score = sc.score(breast_tst)
	print(score)

