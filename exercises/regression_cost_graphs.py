
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


# Ridge Regression (cpu)
cpu_file = "../datasets/cpu/cpu.csv"
cpu = read_csv_file(file=cpu_file, sep=",", features=True, label=True)
cpu.X = StandardScaler().fit_transform(cpu.X)
cpu_trn, cpu_tst = train_test_split(cpu, test_size=0.3, random_state=2)

# Without adaptative alpha
cpu_ridge_1 = RidgeRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=1, adaptative_alpha=False).fit(cpu_trn)
predictions_1 = cpu_ridge_1.predict(cpu_tst)
print("Predictions (cpu ridge regression without adaptative alpha)\n" + str(predictions_1))
cost_dict_1 = cpu_ridge_1.cost_history
plt.plot(list(cost_dict_1.keys()), list(cost_dict_1.values()), linestyle="-", color="red")
plt.title("Ridge Regression without adaptative alpha"), plt.xlabel("Iteration"), plt.ylabel("Cost")
plt.show()

# With adaptative alpha
cpu_ridge_2 = RidgeRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=1, adaptative_alpha=True).fit(cpu_trn)
predictions_2 = cpu_ridge_2.predict(cpu_tst)
print("\nPredictions (cpu ridge regression without adaptative alpha)\n" + str(predictions_2))
cost_dict_2 = cpu_ridge_2.cost_history
plt.plot(list(cost_dict_2.keys()), list(cost_dict_2.values()), linestyle="-", color="blue")
plt.title("Ridge Regression with adaptative alpha"), plt.xlabel("Iteration"), plt.ylabel("Cost")
plt.show()

############################################################################################################################################

# Logistic Regression (breast)
breast_file = "../datasets/breast/breast-bin.csv"
breast = read_csv_file(file=breast_file, sep=",", features=True, label=True)
breast.X = StandardScaler().fit_transform(breast.X)
breast_trn, breast_tst = train_test_split(breast, test_size=0.3, random_state=2)

# Without adaptative alpha
breast_log_1 = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=0.0001, adaptative_alpha=False).fit(breast_trn)
predictions_1 = breast_log_1.predict(breast_tst)
print("\nPredictions (breast logistic regression without adaptative alpha)\n" + str(predictions_1))
cost_dict_1 = breast_log_1.cost_history
plt.plot(list(cost_dict_1.keys()), list(cost_dict_1.values()), linestyle="-", color="red")
plt.title("Logistic Regression without adaptative alpha"), plt.xlabel("Iteration"), plt.ylabel("Cost")
plt.show()

# With adaptative alpha
breast_log_2 = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000, tolerance=0.0001, adaptative_alpha=True).fit(breast_trn)
predictions_2 = breast_log_2.predict(breast_tst)
print("\nPredictions (breast logistic regression without adaptative alpha)\n" + str(predictions_2))
cost_dict_2 = breast_log_2.cost_history
plt.plot(list(cost_dict_2.keys()), list(cost_dict_2.values()), linestyle="-", color="blue")
plt.title("Logistic Regression with adaptative alpha"), plt.xlabel("Iteration"), plt.ylabel("Cost")
plt.show()

