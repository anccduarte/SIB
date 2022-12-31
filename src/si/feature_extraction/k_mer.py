
import itertools
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset

class KMer:

    """
    KMer implements a feature extraction algorithm which computes the normalized frequencies of the
    k-mers of all sequences in a given dataset.
    """

    def __init__(self, k: int = 3, mol_type: str = "dna"):
        """
        Initializes an instance of KMer. KMer implements a feature extraction algorithm which computes
        the normalized frequencies of the k-mers of all sequences in a given dataset.

        Parameters
        ----------
        k: int (default=3)
            The length of the k-mers
        mol_type: str (default="dna")
            Molecular type to be considered when choosing which alphabet to use

        Attributes
        ----------
        fitted: bool
            Whether KMer is already fitted
        alphabet: str
            The alphabet to be used when generating an array of all combinations of length <k>. Depends
            on mol_type (default="dna")
        k_mers: list
            A list containing all possible combinations of k-mers (length <k>)
        """
        # check parameter values
        if k < 1:
            raise ValueError("The value of 'k' must be a positive integer.")
        if mol_type not in ["dna", "protein"]:
            raise ValueError("The value of 'mol_type' must be in {'dna', 'protein'}.")
        # parameters
        self.k = k
        # attributes
        self.fitted = False
        self.alphabet = list("ATCG") if mol_type == "dna" else list("ACDEFGHIKLMNPQRSTVWY")
        self.k_mers = None

    def fit(self, dataset: Dataset) -> "KMer":
        """
        Fits 'KMer' by computing all possible combinations of characters in <self.alphabet> (nucleotides
        or amino acids) of length <self.k>. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        # compute all combinations of size <self.k>
        bases = [self.alphabet for _ in range(self.k)]
        k_mers = itertools.product(*bases)
        # convert tuples to strings
        self.k_mers = ["".join(k_mer) for k_mer in k_mers]
        self.fitted = True
        return self

    def _get_frequencies(self, sequence: np.ndarray) -> np.ndarray:
        """
        Helper method which computes and returns the normalized frequencies of the k-mers present in the
        sequence it takes as a parameter.

        Parameters
        ----------
        sequence: np.ndarray
            The input sequence
        """
        # get k-mers of size <self.k>
        seq = sequence[0].upper()
        seq_kmers = [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
        # get frequencies of each k-mer
        k_mers_dict = {k_mer: 0 for k_mer in self.k_mers}
        for k in seq_kmers:
            k_mers_dict[k] += 1
        # compute and return normalized frequencies
        return np.array([k_mers_dict[kmer] / len(seq) for kmer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms 'KMer' by calculating the normalized frequencies of the k-mers of all sequences in the
        dataset. Returns a new dataset object containing the computed data.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be transformed)
        """
        if not self.fitted:
            raise Warning("Fit KMer before calling 'transform'.")
        # compute normalized frequencies for each sequence in the dataset
        frequencies = np.apply_along_axis(self._get_frequencies, axis=1, arr=dataset.X)
        return Dataset(frequencies, dataset.y, self.k_mers, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits and transforms 'KMeans' by computing all possible combinations of characters in
        <self.alphabet> (nucleotides or amino acids) of length <self.k>, and calculating the normalized
        frequencies of the k-mers of all sequences in the dataset. Returns a new dataset object containing
        the computed data.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (containing the data to be transformed)
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":

    TEST_PATHS = ["../io", "../linear_model", "../model_selection"]
    sys.path.extend(TEST_PATHS)
    from csv_file import read_csv_file
    from logistic_regression import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from split import train_test_split
    
    # TFBS (nucleotide k-mers)
    print("TFBS DATASET (DNA sequences)")
    # read file to Dataset
    path = "../../../datasets/tfbs/tfbs.csv"
    tfbs = read_csv_file(path, sep=",", features=True, label=True)
    # build new Dataset containig the k-mers
    km = KMer()
    tfbs_km = km.fit_transform(tfbs)
    # scale and split
    tfbs_km.X = StandardScaler().fit_transform(tfbs_km.X)
    tfbs_train, tfbs_test = train_test_split(tfbs_km, test_size=0.3, random_state=2)
    # predict labels using LogisticRegression
    lr = LogisticRegression()
    lr.fit(tfbs_train)
    preds = lr.predict(tfbs_test)
    print(f"Predictions:\n{preds}")
    score = lr.score(tfbs_test)
    print(f"Accuracy score: {score*100:.2f}%")

    print("\nTRANSPORTERS DATASET (protein sequences)")
    # TRANSPORTERS (aminoacid k-mers)
    # read csv to Dataset
    path2 = "../../../datasets/transporters/transporters.csv"
    transporters = read_csv_file(path2, sep=",", features=True, label=True)
    # compute new Dataset with k-mer frequencies as features
    km2 = KMer(k=2, mol_type="protein")
    transporters_km = km2.fit_transform(transporters)
    # scale and split
    transporters_km.X = StandardScaler().fit_transform(transporters_km.X)
    transporters_train, transporters_test = train_test_split(transporters_km, 0.3, 2)
    # predict labels using LogisticRegression
    lr2 = LogisticRegression()
    lr2.fit(transporters_train)
    preds2 = lr2.predict(transporters_test)
    print(f"Predictions:\n{preds2}")
    score2 = lr2.score(transporters_test)
    print(f"Accuracy score: {score2*100:.2f}%")

