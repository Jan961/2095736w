import umap
from sklearn.manifold import TSNE
import numpy as np
from Cube import Cube
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from sklearn.decomposition import PCA
from hdimvis.visualise_layouts_and_metrics.plot import show_generation_metrics
from  sklearn.manifold import SpectralEmbedding, MDS
from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from experiments.utils.LayoutHistogram import LayoutHistogram
from experiments.utils.layout_shannon_entropy import calculate_entropy
from hdimvis.metrics.stress.stress import vectorised_stress
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean,manhattan
from pathlib import Path