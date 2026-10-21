import numpy as np
import heapq
from sklearn.cluster import KMeans
from typing import Callable, Tuple, List
from coresets.grid.recursive import RecursiveGridCoreset

class KMeansGuidedRecursiveGrid(RecursiveGridCoreset):
    def __init__(
        self,
        starting_grid_size: float,
        target_coreset_size: int,
        score_fn: Callable[[np.ndarray, np.ndarray], float],
        km_interval: float = 0.1,
        km_kwargs: dict = None,
        km_reuse_centers: bool = False,
    ):
        super().__init__(
            starting_grid_size,
            target_coreset_size,
            score_fn,
        )
        self.km_interval   = km_interval
        self.km_kwargs     = km_kwargs or {}
        self.centers       = None
        self.reuse_centers = km_reuse_centers

    def get_coreset(
        self,
        data: np.ndarray,
        k_clusters: int,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        mins = data.min(axis=0)
        idxs = np.floor((data - mins) / self.starting_grid_size).astype(int)
        buckets = {}
        for pt, idx in zip(data, map(tuple, idxs)):
            buckets.setdefault(idx, []).append(pt)

        heap: List[RecursiveGridCoreset._Cell] = []
        single_cells: List[RecursiveGridCoreset._Cell] = []
        for idx, pts in buckets.items():
            origin = mins + np.array(idx) * self.starting_grid_size
            pts_arr = np.vstack(pts)
            cell    = RecursiveGridCoreset._Cell(origin, self.starting_grid_size, pts_arr, 0)
            if pts_arr.shape[0] == 1:
                single_cells.append(cell)
            else:
                heapq.heappush(heap, cell)

        next_kmeans_run = 0
        total_cells  = len(heap) + len(single_cells)

        while total_cells < self.m:
            next_kmeans_run -= 1
            # every km_interval splits, re-run k-means on **current** coreset
            if next_kmeans_run <= 0:

                pts_list, wts_list = zip(*(c.mean_and_weight() for c in heap + single_cells))
                pts_arr = np.vstack(pts_list)
                wts_arr = np.array(wts_list)

                km_init = self.centers if self.reuse_centers and self.centers is not None else "k-means++"
                km = KMeans(
                    n_clusters=k_clusters,
                    init= km_init,
                    n_init=1,
                    random_state=seed,
                    **self.km_kwargs
                )
                km.fit(pts_arr, sample_weight=wts_arr)
                self.centers = km.cluster_centers_
                next_kmeans_run = self.km_interval * self.m

                # re-score all cells
                for cell in heap:
                    cell.score = self.score_fn(cell.points, self.centers)
                heapq.heapify(heap)

            # pop highest-score cell
            cell = heapq.heappop(heap)

            # split it
            children = cell.split()
            total_cells += len(children) - 1

            # push children with updated scores
            for origin, size, pts in children:
                sc = self.score_fn(pts, self.centers)
                child = RecursiveGridCoreset._Cell(origin, size, pts, sc)
                if pts.shape[0] == 1:
                    single_cells.append(child)
                else:
                    heapq.heappush(heap, child)

            
        pts_out, wts_out = [], []
        for cell in heap + single_cells:
            mu, w = cell.mean_and_weight()
            pts_out.append(mu)
            wts_out.append(w)

        if self.reuse_centers:
            self.kmeans_init = self.centers
        return np.vstack(pts_out), np.array(wts_out)