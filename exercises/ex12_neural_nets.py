
main = "../src/si"
dirs = ["data", "io", "metrics", "model_selection", "neural_networks", "statistics"]
PATHS = [f"{main}/{dir_}" for dir_ in dirs]

import sys
sys.path.extend(PATHS)
import time
from accuracy import accuracy
from csv_file import read_csv_file
from dataset import Dataset
from dense import Dense
from nn import NN
from r2_score import r2_score
from sklearn.preprocessing import StandardScaler
from split import train_test_split


# -- NN REGRESSION (cpu.csv) -> mse / mse_derivative

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


# -- NN BINARY CLASSIFICATION (breast-bin.csv) -> binary_cross_entropy / d_binary_cross_entropy

time.sleep(2)
print("\nNN BINARY CLASSIFICATION (breast-bin)")

# data (breast)
path_breast = "../datasets/breast/breast-bin.csv"
breast = read_csv_file(path_breast, sep=",", features=False, label=True)

# cannot standardize data -> only discrete values

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

# transform predictions to binary
def to_bin(preds):
    mask = preds < 0.5
    preds[mask] = 0
    preds[~mask] = 1

# get accuracy scores (train)
preds_trn = nn_breast.predict(breast_trn)
to_bin(preds_trn)
trn_score_breast = accuracy(breast_trn.y, preds_trn)
print(f"Train score (accuracy): {trn_score_breast:.2%}")

# get accuracy scores (test)
preds_tst = nn_breast.predict(breast_tst)
to_bin(preds_tst)
tst_score_breast = accuracy(breast_tst.y, preds_tst)
print(f"Test score (accuracy): {tst_score_breast:.2%}")

