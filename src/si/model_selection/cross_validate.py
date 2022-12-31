
import numpy as np
import sys
sys.path.append("../data")
from copy import deepcopy
from dataset import Dataset
from split import train_test_split
from typing import Callable


# -- CHECK PARAMETERS

def check_params(dataset: Dataset, cv: int, test_size: float):
    """
    Checks the values of numeric parameters.

    Parameters
    ----------
    cv: int
        The number of folds used in cross-validation
    test_size: float
        The proportion of the dataset to be used for testing
    """
    if cv < 1 or cv > dataset.shape()[0]:
        raise ValueError("The value of 'cv' must be an integer belonging to [1, dim_0(dataset)].")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("The value of 'test_size' must be in (0, 1).")


# -- CROSS-VALIDATE

def cross_validate(model: "estimator",
                   dataset: Dataset,
                   cv: int = 5,
                   random_state: int = None,
                   test_size: float = 0.3,
                   scoring: Callable = None) -> dict:
    """
    Implements a k-fold cross-validation algorithm. Each fold is established randomly. Returns a
    dictionary containing 3 keys:
    1. seeds: The seeds used in the train-test split
    2. train: The scores attained with training data
    3. test: The scores obtained with testing data

    Parameters
    ----------
    model: estimator
        An initialized instance of a classifier/regressor
    dataset: Dataset
        A Dataset object
    cv: int (default=5)
        The number of folds used in cross-validation
    random_state: int (default=None)
        Controls seed generation for splitting the data (allows for reproducible output)
    test_size: float (default=0.3)
        The proportion of the dataset to be used for testing
    scoring: callable (default=None)
        The scoring function used to evaluate the performance of the model (if None, uses the
        model's scoring function)
    """
    # check values of numeric parameters
    check_params(dataset, cv, test_size)
    # main loop -> cross-validate model
    scores = {"seeds": [], "train": [], "test": []}
    for i in range(cv):
        # initilize new instance of 'model' (deepcopy) at each iteration (otherwise, we only have a
        # "fresh" model instance in the first iteration; in subsequent cv iterations, the models would
        # "inherit" the state of the previous iteration)
        Model = deepcopy(model)
        # update random_state so that a distinct seed is generated next time (only if 'int')
        random_state = random_state if random_state is None else random_state + i
        # generate seed for train_test_split and add it to scores
        seed = np.random.RandomState(seed=random_state).randint(0, 100000)
        scores["seeds"] += [seed]
        # split data in train and test
        ds_train, ds_test = train_test_split(dataset=dataset, test_size=test_size, random_state=seed)
        # fit the model on training data
        Model.fit(ds_train)
        # if scoring is None, use the model's scoring function
        if scoring is None:
            train_score = Model.score(ds_train)
            test_score = Model.score(ds_test)
        # otherwise, use the provided scoring function
        else:
            train_score = score(ds_train.y, Model.predict(ds_test))
            test_score = score(ds_test.y, Model.predict(ds_test))
        # add train_score and test_score to scores
        scores["train"] += [train_score]
        scores["test"] += [test_score]
    return scores


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../linear_model"] # "../metrics"
    sys.path.extend(TEST_PATHS)
    # from accuracy import accuracy
    from csv_file import read_csv_file
    from logistic_regression import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    path_to_file = "../../../datasets/breast/breast-bin.csv"
    breast = read_csv_file(file=path_to_file, sep=",", features=False, label=True)
    breast.X = StandardScaler().fit_transform(breast.X)
    
    model = LogisticRegression()
    cv = cross_validate(model=model, dataset=breast, random_state=2) # scoring=accuracy
    print(f"Cross-validation seeds and scores:\n{cv}")
    print(f"Mean score on trainig data: {np.mean(cv['train'])*100:.2f}%")
    print(f"Mean score on testing data: {np.mean(cv['test'])*100:.2f}%")

