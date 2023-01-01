
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset

def one_hot(dataset: Dataset) -> Dataset:
    """
    One-hot encodes the label vector of the Dataset object it takes as input.
    Returns a new Dataset object.

    Parameters
    ----------
    dataset: Dataset
        A Dataset object
    """
    # get one-hot encoded vector
    one_hot_y = []
    for label in dataset.y:
        to_one_hot = []
        for class_ in dataset.get_classes():
            if label == class_:
                to_one_hot.append(1)
            else:
                to_one_hot.append(0)
        one_hot_y.append(to_one_hot)
    # create and return new Dataset object
    one_hot_ds = Dataset(dataset.X, np.array(one_hot_y), dataset.features)
    return one_hot_ds


if __name__ == "__main__":

    print("Original")
    ds = Dataset.from_random(n_examples=20, n_features=10, label=True, seed=2)
    print(ds.label)
    print(ds.y)

    print("\nOne-hot encoded")
    ds_one_hot = one_hot(ds)
    print(ds_one_hot.label)
    print(ds_one_hot.y)

