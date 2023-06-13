import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.utils.get_avg_classwise_f1 import get_avg_classwise_f1
from experiments.utils.get_kmeans_f1 import get_kmeans_f1
from experiments.utils.get_f1_for_best_k_with_knn import get_f1_for_best_k_with_knn
from hdimvis.data_fetchers.DataFetcher import DataFetcher


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

def test_obvious_results():
    dataset = DataFetcher.fetch_data('rna N3k')

    embedding_tsne = TSNE(n_components=2, perplexity=20).fit_transform(dataset.data)
    embedding_PCA = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
    embedding_PCA *= 10 / np.std(embedding_PCA)

    k1 ,f1_pca = get_f1_for_best_k_with_knn(1,20,10,embedding_PCA,dataset.labels)
    k2, f1_tsne = get_f1_for_best_k_with_knn(1, 20, 10, embedding_tsne, dataset.labels)
    print(f"tsne : {f1_tsne}")
    print(f"pca: {f1_pca}")
    assert f1_tsne > f1_pca

    pca =  get_kmeans_f1(embedding_PCA, dataset.labels)
    tsne = get_kmeans_f1(embedding_tsne, dataset.labels)
    print(f"tsne : {tsne}")
    print(f"pca: {pca}")
    assert tsne > pca


