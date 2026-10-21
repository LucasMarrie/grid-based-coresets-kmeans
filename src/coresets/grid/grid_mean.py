from coresets.abstract import AbstractCoreset
import numpy as np


class GridMeanCoreset(AbstractCoreset):

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

    def get_coreset(self, data, k_clusters, seed):
        min_coords = np.min(data, axis=0)
        shifted_data = data - min_coords

        indices = np.floor(shifted_data / self.grid_size).astype(int)

        # cell dict maps cell coordinates to indices of data points
        cell_dict = {}
        for i, coord in enumerate(map(tuple, indices)):
            cell_dict.setdefault(coord, []).append(i)

        coreset_points = []
        weights = []
        for cell_points_idx in cell_dict.values():
            # get the points in the cell
            cell_points = data[cell_points_idx]
            # compute mean and add to coreset
            coreset_point = np.mean(cell_points, axis=0)
            coreset_points.append(coreset_point)
            # cache the cell points for assignments
            weights.append(len(cell_points_idx))

        return np.array(coreset_points), np.array(weights)