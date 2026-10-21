from coresets.abstract import AbstractCoreset

class IdentityCoreset(AbstractCoreset):

    def get_coreset(self, data, k_clusters, seed):
        """
        Identity function for base k-means: returns the full dataset.
        """

        return data

