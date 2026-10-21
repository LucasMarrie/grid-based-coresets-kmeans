from coresets.abstract import AbstractCoreset
import numpy as np

class RandomSamplingCoreset(AbstractCoreset):

    def __init__(self, target_points):
        super().__init__()
        self.target_points = target_points

    def get_coreset(self, data, k_clusters, seed):
        """
        Random sampling coreset method.
        """
        np.random.seed(seed)
        idx = np.random.choice(data.shape[0], self.target_points, replace=False)
        return data[idx]