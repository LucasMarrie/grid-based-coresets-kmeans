import numpy as np
import heapq
from typing import Callable, Tuple, List
from coresets.abstract import AbstractCoreset


class RecursiveGridCoreset(AbstractCoreset):
    def __init__(
        self,
        starting_grid_size: float,
        target_coreset_size: int,
        score_fn: Callable[[np.ndarray], float],
    ):
        """
        Parameters:
        - initial_grid_size: float, width of each grid cell
        - target_coreset_size: int, desired # of coreset points
        - score_fn: function mapping cell_points to score 
        """
        super().__init__()
        self.m        = target_coreset_size
        self.starting_grid_size = starting_grid_size
        self.score_fn = score_fn

    class _Cell:
        def __init__(self, origin: np.ndarray, size: float, points: np.ndarray, score: float):
            self.origin = origin
            self.size   = size
            self.points = points
            self.score  = score

        def __lt__(self, other):
            # we want max-heap behavior via min-heap: larger score = split sooner
            # Therefore negating the score for comparison
            if self.score == other.score:
                return self.points.shape[0] > other.points.shape[0]
            return -self.score < -other.score
        
        def split(self) -> List["RecursiveGridCoreset._Cell"]:
            d    = self.origin.shape[0]
            half = self.size / 2.0
            children = []
            for mask in range(1 << d):
                # build sub-cell origin
                corner = self.origin + half * np.array([((mask>>i)&1) for i in range(d)])
                # select points in that sub-cell
                pts = self.points[
                    np.all((self.points >= corner) & (self.points < corner + half), axis=1)
                ]
                if pts.size > 0:
                    children.append((corner, half, pts))
            return children
        
        def mean_and_weight(self) -> Tuple[np.ndarray,int]:
            return self.points.mean(axis=0), self.points.shape[0]

    def get_coreset(self, data, k_clusters, seed) -> Tuple[np.ndarray,np.ndarray]:
        mins = data.min(axis=0)
        idxs = np.floor((data - mins) / self.starting_grid_size).astype(int)
        cell_buckets = {}
        for pt, idx in zip(data, map(tuple, idxs)):
            cell_buckets.setdefault(idx, []).append(pt)

        heap: List[RecursiveGridCoreset._Cell] = []
        single_cells = []
        for idx, pts in cell_buckets.items():
            origin = mins + np.array(idx) * self.starting_grid_size
            pts_arr = np.vstack(pts)    
            score = self.score_fn(pts_arr)
            cell = RecursiveGridCoreset._Cell(origin, self.starting_grid_size, pts_arr, score)
            if pts_arr.shape[0] == 1:
                single_cells.append(cell)
            else:
                heapq.heappush(heap, cell)

        total_cells = len(heap) + len(single_cells)

        while total_cells < self.m:
            cell = heapq.heappop(heap)

            children = cell.split()
            total_cells += len(children) - 1

            for origin, size, pts in children:
                score = self.score_fn(pts)
                child = RecursiveGridCoreset._Cell(origin, size, pts, score)
                if pts.shape[0] == 1:
                    single_cells.append(child)
                else:
                    heapq.heappush(heap, child)

        pts_out, wts_out = [], []
        for cell in heap + single_cells:
            mu, w = cell.mean_and_weight()
            pts_out.append(mu)
            wts_out.append(w)
        return np.vstack(pts_out), np.array(wts_out)