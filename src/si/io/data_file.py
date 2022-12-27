
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset

def read_data_file(file: str, sep: str, features: bool, label: bool) -> Dataset:
    """
    Reads a text file (e.g., txt, data) and returns a Dataset object.

    Parameters
    ----------
    file: str
        Path to file to be read
    sep: str
        The separator used in the file
    features: bool
        Representative of the presence of a header in the file
    label: bool
        Representative of the presence of a label in the file
    """
    # if the file contains the names of the features, skip_header=1
    ndarr = np.genfromtxt(file, delimiter=sep, skip_header=features)
    # extract the names of the features if they are present in the file
    if features:
        with open(file) as f:
            ncols = f.readline().split(sep)
            ncols[-1] = ncols[-1].strip("\n")
    # initialize X, y, features and label
    if label:
        X = ndarr[:,:-1]
        y = ndarr[:,-1]
        if features:
            *nfeat, nlab = ncols
        else:
            nfeat, nlab = None, None
    else:
        X = ndarr
        nfeat = ncols if features else None
        y, nlab = None, None
    return Dataset(X, y, nfeat, nlab)

def write_data_file(nfile: str, dataset: Dataset, sep: str, label: bool):
    """
    Writes to a txt file from a Dataset object.
    
    Parameters
    ----------
    nfile: str
        Path to file to be written
    dataset: Dataset
        A Dataset object
    sep: str
        The separator to be used in the file
    label: bool
        Representative of the presence of a label in the Dataset object
    """
    header = f"{sep}".join(dataset.features)
    ds = dataset.X
    if label:
        ds = np.concatenate((ds, dataset.y[:, None]), axis=1)
        header += f"{sep}{dataset.label}"
    np.savetxt(nfile, ds, delimiter=sep, header=header, comments="")


if __name__ == "__main__":
    
    # read_data_file
    path1 = "../../../datasets/breast/breast-bin.data"
    ds = read_data_file(file=path1, sep=",", features=False, label=True)
    print(ds.shape(), ds.y.shape)
    # write_data_file
    # path2 = "../../../../../SIB/bin/new_breast.txt"
    # write_data_file(nfile=path2, dataset=ds, sep=" ", label=True)

