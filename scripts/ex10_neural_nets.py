
import numpy as np
import sys
main = "../src/si"
dirs = ["data", "neural_networks", "metrics", "statistics"]
PATHS = [f"{main}/{dir_}" for dir_ in dirs]
sys.path.extend(PATHS)
from activation import identity, relu, sigmoid, softmax
from dataset import Dataset
from dense import Dense
from nn import NN


# -- EXERCISE 10.3

n_ex, n_feat = 64, 32
ds_init = Dataset.from_random(n_ex, n_feat, label=False, seed=2)

# binary -> {0, 1}
y_vec_1 = np.random.RandomState(seed=0).randint(0, 2, n_ex)
ds_1 = Dataset(ds_init.X, y_vec_1)

layers_1 = [Dense(input_size=n_feat, output_size=n_feat, activation="sigmoid"),
            Dense(input_size=n_feat, output_size=n_feat//2, activation="sigmoid"),
            Dense(input_size=n_feat//2, output_size=1, activation="sigmoid")]

nn_model_1 = NN(layers=layers_1)
nn_model_1.fit(ds_1)


# -- EXERCISE 10.4

# multiclass -> {0, 1, 2}
y_vec_2 = np.random.RandomState(seed=0).randint(0, 3, n_ex)
ds_2 = Dataset(ds_init.X, y_vec_2)

layers_2 = [Dense(input_size=n_feat, output_size=n_feat, activation="sigmoid"),
            Dense(input_size=n_feat, output_size=n_feat//2, activation="sigmoid"),
            Dense(input_size=n_feat//2, output_size=3, activation="softmax")]

nn_model_2 = NN(layers=layers_2)
nn_model_2.fit(ds_2)


# -- EXERCISE 10.5

# regression -> uniform distribution over [0, 100)
y_vec_3 = np.random.RandomState(seed=0).rand(n_ex) * 100
ds_3 = Dataset(ds_init.X, y_vec_3)

layers_3 = [Dense(input_size=n_feat, output_size=n_feat, activation="relu"),
            Dense(input_size=n_feat, output_size=n_feat//2, activation="relu"),
            Dense(input_size=n_feat//2, output_size=1, activation="identity")]

nn_model_3 = NN(layers=layers_3)
nn_model_3.fit(ds_3)

