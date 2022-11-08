
import matplotlib.pyplot as plt
import numpy as np
import sys
main = "../src/si"
dirs = ["data", "io", "linear_model", "metrics", "model_selection", "statistics"]
PATHS = [f"{main}/{d}" for d in dirs]
sys.path.extend(PATHS)
from csv_file import read_csv_file
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
from split import train_test_split
from sklearn.preprocessing import StandardScaler


# ridge regression (cpu)
cpu_file = "../datasets/cpu/cpu.csv"
cpu = read_csv_file(file=cpu_file, sep=",", features=True, label=True)
cpu.X = StandardScaler().fit_transform(cpu.X)
cpu_trn, cpu_tst = train_test_split(cpu, test_size=0.3, random_state=2)


# logistic regression (breast)
breast_file = "../datasets/breast/breast-bin.csv"
breast = read_csv_file(file=breast_file, sep=",", features=False, label=True)
breast.X = StandardScaler().fit_transform(breast.X)
breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)


# define variables
n_graphs = 4
datasets = {"cpu": (cpu_trn, cpu_tst), "breast": (breast_trn, breast_tst)}
models = {"cpu": RidgeRegression, "breast": LogisticRegression}
titles = {0: "Ridge Regression without adaptative alpha", 1: "Ridge Regression with adaptative alpha",
          2: "Logistic Regression without adaptative alpha", 3: "Logistic Regression with adaptative alpha"}
l2_penalty = 1
alpha = 0.001
max_iter = 2000
tolerance = (1, 1, 0.0001, 0.0001)
adaptative_alpha = (False, True, False, True)
color = ("red", "blue", "red", "blue")

# main loop -> print predictions and draw graphs for every combination model-alpha
for i in range(n_graphs):
    dataset = "cpu" if i < 2 else "breast"
    model = models[dataset](l2_penalty, alpha, max_iter, tolerance[i], adaptative_alpha[i])
    model.fit(datasets[dataset][0])
    preds = model.predict(datasets[dataset][1])
    print(f"Predictions ({titles[i]})\n{preds}\n")
    cost_dict = model.cost_history
    plt.plot(list(cost_dict.keys()), list(cost_dict.values()), linestyle="-", color=color[i])
    plt.title(titles[i]), plt.xlabel("Iteration"), plt.ylabel("Cost")
    plt.show()

