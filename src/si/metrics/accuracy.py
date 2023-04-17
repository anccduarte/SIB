
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It computes and returns the accuracy score of the model on a given dataset.
    Accuracy score: (TP+TN)/(TP+FP+TN+FN).

    Parameters
    ----------
    y_true: np.ndarray
        The true values of the labels
    y_pred: np.ndarray
        The labels predicted by a classifier
    """
    return np.sum(y_true == y_pred) / len(y_true)

def softmax_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It computes and returns the accuracy score of the model on a given dataset.
    It allows to compute the accuracy score on one-hot encoded vectors.

    Parameters
    ----------
    y_true: np.ndarray
        The one-hot encoded true values of the labels
    y_pred: np.ndarray
        The one-hot encoded labels predicted by a classifier
    """
    # get true and predicted labels
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    # compute and return the accuracy score
    return accuracy(true_labels, pred_labels)


if __name__ == "__main__":
    
    print("ACCURACY")
    true = np.array([0,1,1,1,0,1])
    pred = np.array([1,0,1,1,0,1])
    print(f"Accuracy score: {accuracy(true, pred)*100:.2f}%")

    print("\nSOFTMAX ACCURACY")
    true_s = np.array([[1,0,0], [0,0,1], [0,1,0]])
    pred_s = np.array([[0.65,0.2,0.15], [0.3,0.4,0.3], [0.1,0.7,0.2]])
    print(f"Accuracy score: {softmax_accuracy(true_s, pred_s)*100:.2f}%")

