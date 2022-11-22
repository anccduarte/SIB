
import numpy as np
import pandas as pd

# EX 1.1
iris = pd.read_csv("../datasets/iris/iris.csv")

# EX 1.2
var1 = iris.iloc[:,0]
print("var1 shape")
print(var1.shape) # (150,)

# EX 1.3
iris_tail = iris.tail() # 5, by default
print("\niris tail means")
print(iris_tail.iloc[:,:-1].mean(axis=0))

# EX 1.4
sl, sw, pl, pw = "sepal_length", "sepal_width", "petal_length", "petal_width"
iris_sup1 = iris[(iris[sl] >= 1) & (iris[sw] >= 1) & (iris[pl] >= 1) & (iris[pw] >= 1)]
"""iris_sup1 = np.array(iris.iloc[:,:-1][iris.iloc[:,:-1] >= 1])
idx = np.isnan(iris_sup1).any(axis=1)
iris_sup1 = iris_sup1[~idx,:]"""
print("\niris_sup1 shape")
print(iris_sup1.shape) # (100, 5)

# EX 1.5
setosa = iris[iris["class"] == "Iris-setosa"]
print("\nsetosa shape")
print(setosa.shape) # (50, 5)

