
main = "../src/si"
dirs = ["data", "io", "metrics", "model_selection", "neural_networks", "statistics", "utils"]
PATHS = [f"{main}/{dir_}" for dir_ in dirs]

import numpy as np
import sys
sys.path.extend(PATHS)
import time
from accuracy import accuracy, softmax_accuracy
from csv_file import read_csv_file
from dataset import Dataset
from dense import Dense
from nn import NN
from one_hot import one_hot
from r2_score import r2_score
from sklearn.preprocessing import StandardScaler
from split import train_test_split


# -- NN REGRESSION (cpu.csv) -> mse

print("NN REGRESSION (cpu)")

# data (cpu)
path_cpu = "../datasets/cpu/cpu.csv"
cpu = read_csv_file(path_cpu, sep=",", features=True, label=True)

# standardize data
cpu.X = StandardScaler().fit_transform(cpu.X)

# split data into train and test
cpu_trn, cpu_tst = train_test_split(cpu, random_state=2)

# layers
l1_cpu = Dense(input_size=6, output_size=4, weights_init="ones", activation="relu")
l2_cpu = Dense(input_size=4, output_size=1, activation="identity")
layers_cpu = [l1_cpu, l2_cpu]

# NN model
nn_cpu = NN(layers=layers_cpu, alpha=0.0001, epochs=20000, num_batches=4, verbose=True)
nn_cpu.fit(cpu_trn)
#preds = nn_cpu.predict(cpu_tst)
#print("\nPredictions:")
#print(preds)
trn_score_cpu = nn_cpu.score(dataset=cpu_trn, score_func=r2_score)
print(f"Train score (r2_score): {trn_score_cpu:.2%}")
tst_score_cpu = nn_cpu.score(dataset=cpu_tst, score_func=r2_score)
print(f"Test score (r2_score): {tst_score_cpu:.2%}")


# -- NN BINARY CLASSIFICATION (breast-bin.csv) -> binary cross-entropy

time.sleep(2)
print("\nNN BINARY CLASSIFICATION (breast-bin)")

# data (breast)
path_breast = "../datasets/breast/breast-bin.csv"
breast = read_csv_file(path_breast, sep=",", features=False, label=True)

# standardize data
breast.X = StandardScaler().fit_transform(breast.X)

# split data into train and test
breast_trn, breast_tst = train_test_split(breast, random_state=2)

# layers
l1_breast = Dense(input_size=9, output_size=4, weights_init="ones", activation="relu")
l2_breast = Dense(input_size=4, output_size=1, activation="sigmoid")
layers_breast = [l1_breast, l2_breast]

# NN model
nn_breast = NN(layers=layers_breast,
               alpha=0.0001,
               loss="binary_cross_entropy",
               epochs=10000,
               num_batches=6,
               verbose=True)

# fit model
nn_breast.fit(breast_trn)

# get accuracy scores (train)
trn_score_breast = nn_breast.score(dataset=breast_trn, score_func=accuracy)
print(f"Train score (accuracy): {trn_score_breast:.2%}")

# get accuracy scores (test)
tst_score_breast = nn_breast.score(dataset=breast_tst, score_func=accuracy)
print(f"Train score (accuracy): {tst_score_breast:.2%}")


# -- NN MULTI-CLASS CLASSIFICATION (iris.csv) -> categorical cross-entropy

time.sleep(2)
print("\nNN MULTI-CLASS (iris)")

# data (breast)
path_iris = "../datasets/iris/iris.csv"
iris = read_csv_file(path_iris, sep=",", features=True, label=True)

# standardize data
iris.X = StandardScaler().fit_transform(iris.X)

# one-hot encode iris label
iris = one_hot(iris)

# split data into train and test
# the choice of seed is important as an even distribution of classes in the training data
# allows to attain better results
iris_trn, iris_tst = train_test_split(iris, random_state=2)

# layers
l1_iris = Dense(input_size=4, output_size=4, weights_init="zeros", activation="relu")
l2_iris = Dense(input_size=4, output_size=3, activation="softmax")
layers_iris = [l1_iris, l2_iris]

# NN model
# random_state is redundant in this case (remember it is used to shuffle the data before
# splitting it into batches) as data was suffled in train_test_split
nn_iris = NN(layers=layers_iris,
             alpha=0.0001,
             loss="categorical_cross_entropy",
             epochs=10000,
             num_batches=4,
             random_state=2,
             verbose=True)

# fit model
nn_iris.fit(iris_trn)

# get accuracy scores (train)
trn_score_iris = nn_iris.score(dataset=iris_trn, score_func=softmax_accuracy)
print(f"Train score (accuracy): {trn_score_iris:.2%}")

# get accuracy scores (test)
tst_score_iris = nn_iris.score(dataset=iris_tst, score_func=softmax_accuracy)
print(f"Test score (accuracy): {tst_score_iris:.2%}")

