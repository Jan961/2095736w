from sklearn.decomposition import PCA

from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layout
import numpy as np
import matplotlib.pyplot as plt

#
# coil = dataset = DataFetcher.fetch_data('coil20')
# embedding_PCA = PCA(n_components=2, whiten=False, copy=True).fit_transform(coil.data).astype(np.float64)
# embedding_PCA *= 10/np.std(embedding_PCA)
#
# show_layout(positions=embedding_PCA, labels=coil.labels)
rna_dataset = DataFetcher.fetch_data('cancer RNA')
