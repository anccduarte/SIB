
import numpy as np
import sys
PATHS = ["../data", "../metrics", "../statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from accuracy import accuracy
from sigmoid_function import sigmoid_function
from typing import Union

class LogisticRegression:

    """
    LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.
    """

    def __init__(self,
                 l2_penalty: Union[int, float] = 1,
                 alpha: Union[int, float] = 0.001,
                 max_iter: int = 1000,
                 tolerance: Union[int, float] = 0.0001,
                 adaptative_alpha: bool = False):
        """
        LogisticRegression is a logistic model using the L2 regularization.
        This model solves the logistic regression problem using an adapted Gradient Descent technique.

        Parameters
        ----------
        l2_penalty: int, float (default=1)
            The L2 regularization parameter
        alpha: int, float (default=0.001)
            The learning rate
        max_iter: int (default=1000)
            The maximum number of iterations
        tolerance: int, float (default=0.0001)
            Tolerance for stopping gradient descent
        adaptative_alpha: bool (default=False)
            Whether an adaptative alpha is used in the gradient descent

        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        theta: np.ndarray
            Model parameters, namely the coefficients of the linear model.
            For example, x0 * theta[0] + x1 * theta[1] + ...
        theta_zero: float
            Model parameter, namely the intercept of the linear model.
            For example, theta_zero * 1
        cost_history: dict
            A dictionary containing the values of the cost function (J function) at each iteration
            of the algorithm (gradient descent)     
        """
        # check values
        if l2_penalty <= 0:
            raise ValueError("The value of 'l2_penalty' must be positive.")
        if alpha <= 0:
            raise ValueError("The value of 'alpha' must be positive.")
        if max_iter < 1:
            raise ValueError("The value of 'max_iter' must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("The value of 'tolerance' must be positive.")
        # parameters
        self.l2_penalty = l2_penalty # lambda
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.adaptative_alpha = adaptative_alpha
        # attributes
        self.fitted = False
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def _gradient_descent_iter(self, dataset: Dataset, m: int) -> None:
        """
        Performs one iteration of the gradient descent algorithm. The algorithm goes as follows:
        1. Predicts the outputs of the dataset
            -> X @ theta + theta_zero (it then uses the sigomid function)
        2. Computes the gradient vector and adjusts it according to the value of alpha
            -> (alpha / m) * (y_pred - y_true) @ X
        3. Computes the penalization term
            -> theta * alpha * (l2 / m)
        4. Updates theta
            -> theta = theta - gradient - penalization
        5. Updates theta_zero
            -> theta_zero = theta_zero - (alpha / m) * SUM[y_pred - y_true] * X(0), X(0) = [1,1,...,1]

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        m: int
            The number of examples in the dataset
        """
        # predicted y (uses the sigmoid function)
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        # computing the gradient vector given a learning rate alpha
        # vector of shape (n_features,) -> gradient[k] updates self.theta[k]
        gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, dataset.X)
        # computing the penalization term
        penalization_term = self.theta * self.alpha * (self.l2_penalty / m)
        # updating the model parameters (theta and theta_zero)
        self.theta = self.theta - gradient - penalization_term
        self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - dataset.y)

    def _regular_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. Does not update the learning rate (self.alpha). Covergence is attained 
        whenever the difference of cost function values between iterations is less than self.tolerance.
        Returns self (fitted model).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        # get the shape of the dataset
        m, n = dataset.shape()
        # initialize the model parameters (it can be initialized randomly using a range of values)
        self.theta = np.zeros(n)
        self.theta_zero = 0
        # main loop -> gradient descent
        i = 0
        converged = False
        while i < self.max_iter and not converged:
            # compute gradient descent iteration (update model parameters)
            self._gradient_descent_iter(dataset, m)
            # add new entry to self.cost_history
            self.cost_history[i] = self.cost(dataset)
            # verify convergence
            converged = abs(self.cost_history[i] - self.cost_history.get(i-1, np.inf)) < self.tolerance
            i += 1
        return self

    def _adaptative_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. Updates the learning rate (self.alpha), by halving it every
        time the difference of cost function values between iterations is less than self.tolerance.
        Returns self (fitted model).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        # get the shape of the dataset
        m, n = dataset.shape()
        # initialize the model parameters (it can be initialized randomly using a range of values)
        self.theta = np.zeros(n)
        self.theta_zero = 0
        # main loop -> gradient descent
        for i in range(self.max_iter):
            # compute gradient descent iteration (update model parameters)
            self._gradient_descent_iter(dataset, m)
            # add new entry to self.cost_history
            self.cost_history[i] = self.cost(dataset)
            # update learning rate
            is_lower = abs(self.cost_history[i] - self.cost_history.get(i-1, np.inf)) < self.tolerance
            if is_lower: self.alpha /= 2
        return self

    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. If self.adaptative_alpha is True, fits the model by updating the
        learning rate (alpha). Returns self (fitted model).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        self.fitted = True
        return self._adaptative_fit(dataset) if self.adaptative_alpha else self._regular_fit(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset. Applies the sigmoid function, returning 0 if
        y_pred < 0.5, otherwise 1.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to predict the output of)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'predict'.")
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        mask = y_pred >= 0.5
        y_pred[~mask], y_pred[mask] = 0, 1
        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy score of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the accuracy on)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'score'.")
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes and returns the value of the cost function (J function) of the model on the dataset
        using L2 regularization.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the cost function on)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'cost'.")
        m = dataset.shape()[0]
        # compute actual predictions (not 'binarized')
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        # calculate value of the cost function
        first_sum = -np.sum(dataset.y * np.log(predictions) + (1-dataset.y) * np.log(1-predictions)) / m
        second_sum = (self.l2_penalty / (2*m)) * np.sum(self.theta**2)
        cost = first_sum + second_sum
        return cost


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../model_selection"]
    sys.path.extend(TEST_PATHS)
    from csv_file import read_csv_file
    from sklearn.preprocessing import StandardScaler
    from split import train_test_split

    path_to_file = "../../../datasets/breast/breast-bin.csv"
    breast = read_csv_file(file=path_to_file, sep=",", features=False, label=True)
    breast.X = StandardScaler().fit_transform(breast.X)
    breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)
    breast_logistic = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=0.0001, adaptative_alpha=False)
    breast_logistic = breast_logistic.fit(breast_trn)
    predictions = breast_logistic.predict(breast_tst)
    score = breast_logistic.score(breast_tst)
    print(f"Predictions:\n{predictions}")
    print(f"\nScore: {score*100:.2f}%")

