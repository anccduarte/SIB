
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset

def read_data_file(file: str, sep: str, features: bool, label: bool):
	"""
	Lê um ficheiro de texto (p.e., txt, data) e retorna um objeto Dataset.
	"""
	# caso o ficheiro contenha o nome das features, skip_header=1
	ndarr = np.genfromtxt(file, delimiter=sep, skip_header=features)
	# extraír nomes das features caso estejam presentes
	if features:
		with open(file) as f:
			ncols = f.readline().split(sep)
			ncols[-1] = ncols[-1].strip("\n")
	# atribuír valores aos atributos X, y, features, label do objeto Dataset
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
	Gera um ficheiro txt a partir de um objeto Dataset.
	"""
	header = " ".join(dataset.features)
	ds = dataset.X
	if label:
		ds = np.concatenate((ds, dataset.y[:,None]), axis=1)
		header += f" {dataset.label}"
	np.savetxt(nfile, ds, delimiter=sep, header=header, comments="")


if __name__ == "__main__":
	# read_data_file
	path1 = "../../../datasets/breast/breast-bin.data"
	ds = read_data_file(file=path1, sep=",", features=False, label=True)
	print(ds.shape())
	# write_data_file
	path2 = "../../../../../bin/new_breast.txt"
	write_data_file(nfile=path2, dataset=ds, sep=" ", label=True)

