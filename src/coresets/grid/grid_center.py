from coresets.abstract import AbstractCoreset
import numpy as np

class GridCenterCoreset(AbstractCoreset):

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

    def get_coreset(self, data, k_clusters, seed):
        min_coords = np.min(data, axis=0)
        shifted_data = data - min_coords

        indices = np.floor(shifted_data / self.grid_size).astype(int)

        cell_dict = {}
        for i, coord in enumerate(map(tuple, indices)):
            cell_dict.setdefault(coord, []).append(i)

        coreset_points = []
        weights = []
        for idx, cell_points_idx in cell_dict.items():
            mid_point = min_coords + (np.array(idx) + 0.5) * self.grid_size
            coreset_points.append(mid_point)
            weights.append(len(cell_points_idx))
    
        return np.array(coreset_points), np.array(weights)
    