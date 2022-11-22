
import numpy as np
import sys
PATHS = ["../src/si/data", "../src/si/neural_networks"]
sys.path.extend(PATHS)
from activation import identity, relu, sigmoid, softmax
from dataset import Dataset
from dense import Dense
from nn import NN


# Exercise 10.3

n_ex, n_feat = 64, 32
ds_init = Dataset.from_random(n_ex, n_feat, label=False, seed=2)

# binary -> {0, 1}
y_vec_1 = np.random.RandomState(seed=0).randint(0, 2, n_ex)
ds_1 = Dataset(ds_init.X, y_vec_1)

layers_1 = [Dense(n_feat, n_feat//2, activation_function=sigmoid),
            Dense(n_feat//2, n_feat//4, activation_function=sigmoid),
            Dense(n_feat//4, n_feat//8, activation_function=sigmoid)]

nn_model_1 = NN(layers=layers_1)
nn_model_1.fit(ds_1)


# Exercise 10.4

# multiclass -> {0, 1, 2}
y_vec_2 = np.random.RandomState(seed=0).randint(0, 3, n_ex)
ds_2 = Dataset(ds_init.X, y_vec_2)

layers_2 = [Dense(n_feat, n_feat//2, activation_function=sigmoid),
            Dense(n_feat//2, n_feat//4, activation_function=sigmoid),
            Dense(n_feat//4, n_feat//8, activation_function=softmax)]

nn_model_2 = NN(layers=layers_2)
nn_model_2.fit(ds_2)


# Exercise 10.5

# regression -> uniform distribution over [0, 100)
y_vec_3 = np.random.RandomState(seed=0).rand(n_ex) * 100
ds_3 = Dataset(ds_init.X, y_vec_3)

layers_3 = [Dense(n_feat, n_feat//2, activation_function=relu),
            Dense(n_feat//2, n_feat//4, activation_function=relu),
            Dense(n_feat//4, n_feat//8, activation_function=identity)]

nn_model_3 = NN(layers=layers_3)
nn_model_3.fit(ds_3)

