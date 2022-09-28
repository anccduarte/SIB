
import numpy as np
import pandas as pd
import sys
sys.path.append("../data")
from dataset import Dataset

# não utiliza o argumento 'features'
def read_csv_file(file: str, sep: str, label: bool):
	"""
	Lê um ficheiro csv e retorna um objeto Dataset.
	"""
	df = pd.read_csv(file, sep=sep)
	col_names = list(df.columns)
	if label:
		X = np.array(df.iloc[:,:-1])
		y = np.array(df.iloc[:,-1])
		*name_feats, name_lab = col_names
	else:
		X = np.array(df)
		name_feats = col_names
		y, name_lab = None, None
	return Dataset(X, y, name_feats, name_lab)

# não utiliza o argumento 'features'
def write_csv_file(nfile: str, dataset: Dataset, sep: str, label: bool):
	"""
	Gera um ficheiro csv a partir de um objeto Dataset.
	"""
	df = pd.DataFrame(dataset.X)
	df.columns = dataset.features
	if label:
		df = pd.concat([df, pd.Series(dataset.y)], axis=1)
		df.columns = dataset.features + [dataset.label]
	df.to_csv(nfile, sep)
	

if __name__ == "__main__":
	# read_csv
	path1 = "../../../datasets/iris/iris.csv"
	ds = read_csv_file(file=path1, sep=",", label=True)
	print(ds.X.shape, ds.y.shape)
	# write_csv
	path2 = "~/Downloads/bin/new_iris.csv"
	write_csv_file(nfile=path2, dataset=ds, sep=",", label=True)

