
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset

class PCA:

    """
    PCA implementation to reduce the dimensions of a given dataset. It uses SVD (Singular
    Value Decomposition) to do so.
    """

    def __init__(self, n_components: int = 10):
        """
        PCA implementation to reduce the dimensions of a given dataset. It uses SVD (Singular
        Value Decomposition) to do so.

        Parameters
        ----------
        n_components: int (default=10)
            The number of principal components to be computed

        Attributes
        ----------
        fitted: bool
            Whether 'PCA' is already fitted
        mean: np.ndarray
            The mean value of each feature of the dataset
        components: np.ndarray
            The first <n_components> principal components
        explained_variance: np.ndarray
            The variances explained by the first <n_components> principal components
        """
        # parameters
        if n_components < 1:
            raise ValueError("The value of 'n_components' must be greater than 0.")
        self.n_components = n_components
        # attributes
        self.fitted = False
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> "PCA":
        """
        Fits PCA by computing the mean value of each feature of the dataset, the first 
        <n_components> principal components and the corresponding explained variances.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        # center data (X - mean)
        self.mean = np.mean(dataset.X, axis=0)
        X_centered = dataset.X - self.mean
        # calculate SVD -> X_centered = U@np.diag(S)@V_t
        U, S, V_t = np.linalg.svd(X_centered, full_matrices=False)
        # determine the first <n_components> components
        self.components = V_t[:self.n_components]
        # infer explained variances
        n = dataset.shape()[0]
        variances = (S**2) / (n - 1)
        self.explained_variance = variances[:self.n_components]
        # update fitted
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transforms the dataset by reducing X (X_reduced = X * V, V = self.components.T).
        Returns X reduced.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        if not self.fitted:
            raise Warning("Fit 'PCA' before calling 'transform'.")
        # center data
        X_centered = dataset.X - self.mean
        # calculate X reduced
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Fits PCA and transforms the dataset by reducing X. Returns X reduced.
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":

    sys.path.append("../io")
    from csv_file import read_csv_file

    print("EX1")
    ds = Dataset.from_random(n_examples=10, n_features=10, label=False, seed=0)
    pca = PCA(n_components=4)
    pca.fit(ds)
    #print(pca.mean)
    #print(pca.components)
    #print(pca.explained_variance)
    x_reduced = pca.transform(ds)
    print(x_reduced)
    
    print("\nEX2 - iris")
    path = "../../../datasets/iris/iris.csv"
    iris = read_csv_file(file=path, sep=",", features=True, label=True)
    pca_iris = PCA(n_components=2)
    iris_reduced = pca_iris.fit_transform(iris)
    print(iris_reduced)
    
