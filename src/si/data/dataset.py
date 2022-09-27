
import numpy as np
import pandas as pd

class Dataset:

	def __init__(self, X, y=None, features=None, label=None):
		"""
		Inicializa uma instância da classe Dataset.
		"""
		if type(X) is not np.ndarray:
			raise TypeError("O parâmetro 'X' deve ser do tipo 'np.ndarray'.")
		if type(y) is not np.ndarray and y is not None:
			raise TypeError("O parâmetro 'y' deve ser do tipo 'np.ndarray' ou 'None'.")
		if type(features) not in (list, tuple) and features is not None:
			raise TypeError("O parâmetro 'features' deve ser do tipo 'list', 'tuple' ou 'None'.")
		if type(label) is not str and label is not  None:
			raise TypeError("O parâmetro 'label' deve ser do tipo 'str' ou 'None'.")
		if y is not None:
			if X.shape[0] != y.size:
				raise ValueError("O número de exemplos de 'X' deve ser igual à dimensão de 'y'.")
		self.X = X
		self.y = y
		self.features = [f"feat{i+1}" for i in range(X.shape[1])] if features is None else features
		self.label = "label" if label is None else label

	def shape(self):
		"""
		Retorna um tuplo contendo as dimensões do dataset.
		"""
		return self.X.shape

	def has_label(self):
		"""
		Retorna um valor booleano indicativo da presença de labels.
		"""
		return True if self.y is not None else False

	def get_classes(self):
		"""
		Retorna um np.ndarray contendo as classes do dataset.
		"""
		return np.unique(self.y)
		#return list(set(self.y))

	def get_mean(self):
		"""
		Retorna um np.ndarray contendo a média de cada coluna do dataset.
		"""
		return self.X.mean(axis=0)

	def get_variance(self):
		"""
		Retorna um np.ndarray contendo a variância de cada coluna do dataset.
		"""
		return self.X.var(axis=0)

	def get_median(self):
		"""
		Retorna um np.ndarray contendo a mediana de cada coluna do dataset.
		"""
		return np.median(self.X, axis=0)

	def get_min(self):
		"""
		Retorna um np.ndarray contendo o valor mínimo de cada coluna do dataset.
		"""
		return self.X.min(axis=0)

	def get_max(self):
		"""
		Retorna um np.ndarray contendo o valor máximo de cada coluna do dataset.
		"""
		return self.X.max(axis=0)

	def summary(self):
		"""
		Retorna um pd.Dataframe contendo métricas descritivas (média, variância, mediana, mínimo
		e máximo) de cada coluna do dataset.
		"""
		df = pd.DataFrame({
			"Mean": self.get_mean(),
			"Variance": self.get_variance(),
			"Median": self.get_median(),
			"Min": self.get_min(),
			"Max": self.get_max()
			})
		return df

	def remove_nan(self):
		"""
		Remove linhas do dataset que contenham valores omissos (NaN).
		"""
		idx = np.isnan(self.X).any(axis=1)
		self.X, self.y = self.X[~idx], self.y[~idx]

	def fill_nan(self, fill):
		"""
		Subtitui todos os valores omissos (NaN) do dataset pela média/mediana da respetiva coluna.
		"""
		vals = {"mean": np.nanmean, "median": np.nanmedian}
		self.X = np.nan_to_num(self.X, nan=vals[fill](self.X,axis=0))

if __name__ == "__main__":

	print("EX1")
	X1 = np.array([[1,2,3],[4,5,6]])
	y1 = np.array([1,2])
	f1 = ["A","B"]
	l1 = "y"
	ds1 = Dataset(X1, y1, f1, l1)
	print(f"shape: {ds1.shape()}")
	print(f"has_label: {ds1.has_label()}")
	print(f"classes: {ds1.get_classes()}")
	print(ds1.summary())

	print("\nEX2")
	X2 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
	y2 = np.array([7,8,9])
	ds2 = Dataset(X2, y2)
	print(f"shape (before removing NaNs): {ds2.shape()}")
	ds2.remove_nan()
	print(f"shape (after removing NaNs): {ds2.shape()}")
	print(ds2.summary())

	print("\nEX3")
	X3 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
	y3 = np.array([7,8,9])
	ds3 = Dataset(X3, y3)
	print(f"shape (before filling NaNs): {ds3.shape()}")
	ds3.fill_nan("mean")
	print(f"shape (after filling NaNs): {ds3.shape()}")
	print(ds3.summary())

