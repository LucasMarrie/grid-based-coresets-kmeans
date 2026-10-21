import numpy as np


def binary_search_grid_size(
    data: np.ndarray,
    target_cells: int,
    tol: float = 0.01,
    max_iter: int = 30,
    sample_size: int = None,
    random_state: int = 42,
    expand_factor: float = 2.0,
    max_expand: int = 10
) -> float:
    """
    Approximate a grid cell width g so that binning data into cells of width g
    produces target_cells nonempty cells, to within tol fraction.
    Automatically computes and expands volume- and variance-based bounds.

    Parameters
    ----------
    data : (n, d) np.ndarray
    target_cells : int
        Desired number of nonempty cells.
    tol : float
        Relative tolerance on cell count (e.g. 0.05 for ±5%).
    max_iter : int
        Max binary-search iterations.
    sample_size : int or None
        If not None and n > sample_size, downsample for speed.
    random_state : int
        Seed for sampling.
    expand_factor : float
        Factor to expand bounds when bracketing.
    max_expand : int
        Max expansions when attempting to bracket.

    Returns
    -------
    g : float
        Approximate grid cell width.
    """
    rng = np.random.default_rng(random_state)
    n, d = data.shape

    # Optional downsampling for speed
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        samp = data[idx]
    else:
        samp = data

    # volume-based lower bound g_lo
    mins = samp.min(axis=0)
    maxs = samp.max(axis=0)
    lengths = maxs - mins
    volume = np.prod(lengths)
    g_lo = (volume / target_cells) ** (1.0 / d)

    # variance-based upper bound g_hi
    cov = np.cov(samp, rowvar=False)
    g_hi = np.sqrt(np.trace(cov) / d)

    # count nonempty cells at width g
    def count_cells(g: float) -> int:
        idxs = np.floor((samp - mins) / g).astype(int)
        return len({tuple(row) for row in idxs})

    # bracket so that count(g_lo) >= target >= count(g_hi)
    for _ in range(max_expand):
        c_lo = count_cells(g_lo)
        c_hi = count_cells(g_hi)
        if c_lo >= target_cells >= c_hi:
            break
        if c_lo < target_cells:
            g_lo /= expand_factor
        if c_hi > target_cells:
            g_hi *= expand_factor
    else:
        # If we never bracketed, raise an error
        raise RuntimeError(
            f"Could not bracket target={target_cells}: "
            f"cells@lo={c_lo}, cells@hi={c_hi}"
        )

    # binary search within [g_lo, g_hi]
    lo, hi = g_lo, g_hi
    best_g, best_diff = lo, float('inf')

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m_mid = count_cells(mid)
        diff = abs(m_mid - target_cells)

        # remember closest
        if diff < best_diff:
            best_diff, best_g = diff, mid

        # stop if within tolerance
        if diff <= tol * target_cells:
            return mid

        # adjust bracket
        if m_mid > target_cells:
            lo = mid  # too many cells 
        else:
            hi = mid  # too few cells 

    return best_g