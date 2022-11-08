
import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Euclidean distances between a given vector x and all vectors in an 
    array of vectors y. Returns a 1-dimensional vector containing the computed distances.
    Euclidean distance: sqrt(SUM[(pi - qi)^2])

    Parameters
    ----------
    x: np.ndarray
        An array consisting of one row
    y: np.ndarray
        An array containing one or multiple rows
    """
    # vectorized operation: each data point in x is compared to each data point in y; also,
    # since y has n rows, this operation is performed n times
    # axis=1 -> the comparisons are made columnwise
    distances = np.sqrt(np.square(x-y).sum(axis=1))
    return distances

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Manhattan distances between a given vector x and all vectors in an 
    array of vectors y. Returns a 1-dimensional vector containing the computed distances.
    Manhattan distance: SUM[abs(pi - qi)]

    Parameters
    ----------
    x: np.ndarray
        An array consisting of one row
    y: np.ndarray
        An array containing one or multiple rows
    """
    distances = np.absolute(x-y).sum(axis=1).astype("float")
    return distances


if __name__ == "__main__":

    import sys
    sys.path.append("../data")
    from dataset import Dataset

    ds = Dataset.from_random(n_examples=10, n_features=10, label=False, seed=0)
    x = ds.X[0,:]
    y = ds.X[1:,]
    print(euclidean_distance(x,y))
    print(manhattan_distance(x,y))

