import faiss
import numpy as np
from faiss import METRIC_INNER_PRODUCT


def cluster(x: np.ndarray, num_clusters: int, num_iterations=10, verbose=False):
    verbose = True
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=num_iterations, verbose=verbose)
    kmeans.train(x)
    distance, index = kmeans.assign(x)
    return index


def get_cosine_distance_matrix(m, n, normalize=False):
    if normalize:
        faiss.normalize_L2(m)
        faiss.normalize_L2(n)
    distance_matrix = faiss.pairwise_distances(m, n, metric_arg=METRIC_INNER_PRODUCT)
    return distance_matrix
