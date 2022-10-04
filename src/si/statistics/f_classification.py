
import numpy as np
import sys
sys.path.append("../data")
from dataset import Dataset
from scipy import stats
from typing import Tuple

def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Scoring function for classification problems. It computes one-way ANOVA F-scores for the
    provided dataset. The F-scores allow to analyze if the mean between two or more groups
    (factors) are significantly different. Samples are grouped by the labels of the dataset.
    Returns a tuple (F, p) of np.ndarrays, such that F contains the F-scores of the features 
    and p consists of the p-values of the F-scores.

    Parameters
    ----------
    dataset: Dataset
    	A labeled Dataset object
	"""
	classes = dataset.get_classes()
	groups = [dataset.X[dataset.y == c] for c in classes]
	F, p = stats.f_oneway(*groups)
	return F, p


if __name__ == "__main__":
	# 3 columns => len(F) = len(p) = 3
	# each column of each class is compared to the respetive columns of the remaining classes (???)
	# i.e., [1,4] is compared to [7,4], [2,5] to [8,3], and [3,6] to [9,7]
	ds = Dataset(np.array([[1,2,3],[4,5,6],[7,8,9],[4,3,7]]), np.array([1,1,0,0]))
	# ds = Dataset(np.array([[1],[4],[7],[4]]), np.array([1,1,0,0]))
	print(f_classification(ds))

