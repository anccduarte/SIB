
import numpy as np
import pandas as pd
import sys
sys.path.append("../data")
from dataset import Dataset

def read_csv_file(file: str, sep: str, features: bool, label: bool) -> Dataset:
	"""
	Reads a csv file and returns a Dataset object.
	
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
	df = pd.read_csv(file, sep=sep, header="infer" if features else None)
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

def write_csv_file(nfile: str, dataset: Dataset, sep: str, label: bool):
	"""
	Writes to a csv file from a Dataset object.

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
	df = pd.DataFrame(dataset.X)
	df.columns = dataset.features
	if label:
		df = pd.concat([df, pd.Series(dataset.y)], axis=1)
		df.columns = dataset.features + [dataset.label]
	df.to_csv(nfile, sep)
	

if __name__ == "__main__":
	# read_csv
	path1 = "../../../datasets/iris/iris.csv"
	ds = read_csv_file(file=path1, sep=",", features=True, label=True)
	print(ds.X.shape, ds.y.shape)
	# write_csv
	path2 = "~/Downloads/SIB/bin/new_iris.csv"
	write_csv_file(nfile=path2, dataset=ds, sep=",", label=True)

