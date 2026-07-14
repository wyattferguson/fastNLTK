"""
fastnltk.cluster — Drop-in replacement for nltk.cluster.

Rust-accelerated KMeansClusterer with Euclidean distance.
Other clusterers (GAAC, VectorSpace) fall back to NLTK.
"""

from nltk.cluster import *  # noqa: F403

from fastnltk._rust import KMeansClusterer as _RustKMeansClusterer


class KMeansClusterer:
    """Rust-accelerated K-means clustering."""
    def __init__(self, num_clusters, max_iterations=50):
        self._impl = _RustKMeansClusterer(num_clusters, max_iterations)

    def cluster(self, vectors):
        return self._impl.cluster(vectors)

    def classify(self, vector):
        return self._impl.classify(vector)

    def centroids(self):
        return self._impl.centroids()

    def labels(self):
        return self._impl.labels()
