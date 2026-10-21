class AbstractCoreset:

    def __init__(self, **kwargs):
        self.kmeans_init = 'k-means++'

    def get_coreset(self, data, k_clusters, seed):
        raise NotImplementedError("Subclasses must implement this method.")
    