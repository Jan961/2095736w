import numpy as np

from experiments.utils.get_avg_classwise_f1 import get_avg_classwise_f1
from experiments.utils.get_kmeans_f1 import get_kmeans_f1
from experiments.utils.get_f1_for_best_k_with_knn import get_f1_for_best_k_with_knn


true_labels = np.array([0,0,1,1,2,2,3,])
predicted = np.array([1,1,0,0,1,1,1])
mock_data1 = np.array([[-2,-2],[-1.8,-1.8],[-2,2],[-1.8,1.8],[2,2],[1.8,1.8],[2,-2]])

def test_classwise_f1():
    assert get_avg_classwise_f1(true_labels, true_labels) == 1
    assert get_avg_classwise_f1(true_labels,predicted) == 0

def test_kmeans_f1():
    assert get_kmeans_f1(mock_data1,true_labels) == 1
    assert get_kmeans_f1(np.random.randint(1,10,(7,2)), true_labels) < 1

def test_f1_for_best_k_with_knn():
    k, f1 = get_f1_for_best_k_with_knn(1,4,2,mock_data1,true_labels)
    assert k <= 2
    assert f1 < 1



