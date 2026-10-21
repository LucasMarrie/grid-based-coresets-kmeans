import numpy as np
from scipy.stats import entropy

"""
Recursive grid search heuristics for scoring cells.
"""

def population_score(cell_points: np.ndarray) -> float:
    """
    Score by the number of points in the cell.
    """
    return float(cell_points.shape[0])


# 2. Variance: split the cells with greatest spread
def variance_score(cell_points: np.ndarray) -> float:
    """
    Score by sum of coordinate-wise variances.
    """
    return float(np.sum(np.var(cell_points, axis=0)))


def inertia_score(cell_points: np.ndarray) -> float:
    """
    Sum of squared distances from each point in the cell to the cell's centroid.
    """
    n = cell_points.shape[0]
    if n < 2:
        return 0.0
    mu = cell_points.mean(axis=0)
    diffs = cell_points - mu
    return float(np.sum(np.einsum('ij,ij->i', diffs, diffs)))


def mad_score(cell_points: np.ndarray) -> float:
    """
    Mean absolute distance to the cell's centroid.
    """
    n = cell_points.shape[0]
    if n == 0:
        return 0.0
    mu = cell_points.mean(axis=0)
    return float(np.mean(np.linalg.norm(cell_points - mu, axis=1)))

def diameter_score(cell_points: np.ndarray) -> float:
    """
    Maximum Euclidean distance between any two points in the cell.
    """
    n, d = cell_points.shape
    if n < 2:
        return 0.0

    # Approximate diameter via centroid farthest-point:
    mu = cell_points.mean(axis=0)
    dists = np.linalg.norm(cell_points - mu, axis=1)
    return float(2.0 * dists.max())





"""
Kmeans guided recursive grid search heuristics.
"""
def impurity_score(cell_points: np.ndarray, centers: np.ndarray) -> float:
    """
    Score by the fraction of points NOT in the largest-assigned cluster
    """
    n = cell_points.shape[0]
    if n == 0:
        return 0.0

    d2 = np.linalg.norm(cell_points[:, None] - centers, axis=2)**2
    labels = np.argmin(d2, axis=1)

    counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
    max_count = counts.max()

    return float(1.0 - max_count / n)


def error_score(cell_pts, centers):
    d2 = np.linalg.norm(cell_pts[:,None] - centers, axis=2)**2
    return float(np.sum(np.min(d2, axis=1)))


def margin_score(cell_pts, centers):
    d2 = np.linalg.norm(cell_pts[:,None] - centers, axis=2)**2
    sorted_d2 = np.sort(d2, axis=1)
    gaps = np.sqrt(sorted_d2[:,1] - sorted_d2[:,0])
    return float(-np.mean(gaps))  # negative so smaller gaps = higher priority


def label_entropy_score(pts, centers):
    d2 = np.linalg.norm(pts[:,None]-centers,axis=2)**2
    labels = np.argmin(d2,axis=1)
    counts = np.bincount(labels, minlength=centers.shape[0]).astype(float)
    probs = counts / counts.sum()
    H = entropy(probs)             
    return float(H / np.log(len(centers)))