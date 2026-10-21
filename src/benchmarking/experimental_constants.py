"""
Experimental constants for the full benchmarking suite. (Not for prelimanry tests)
"""

from benchmarking.datasets import (
    get_blobs_2d,
    get_blobs_3d,
    get_covertype_data,
    get_us_census_data,
)
from coresets.grid.score_heuristics import (
    population_score,
    variance_score,
    inertia_score,
    diameter_score,
    mad_score,
    impurity_score,
    error_score,
    label_entropy_score,
    margin_score
)

covertype_2d, covertype_3d = get_covertype_data()
us_census_2d, us_census_3d = get_us_census_data()

ms = [50, 100, 200, 500]
ks = [10, 20, 30, 40, 50]

test_datasets = []
for k in ks:
    test_datasets += [
        {"name": f"2d blobs", "dataset": get_blobs_2d(1000000, k), "k_clusters": k},
        {"name": f"3d blobs", "dataset": get_blobs_3d(1000000, k), "k_clusters": k},
        {"name": f"Covertype 2d", "dataset": covertype_2d, "k_clusters": k},
        {"name": f"Covertype 3d", "dataset": covertype_3d, "k_clusters": k},
        {"name": f"US Census 2d", "dataset": us_census_2d, "k_clusters": k},
        {"name": f"US Census 3d", "dataset": us_census_3d, "k_clusters": k},
    ]

starting_factors = [0.1, 0.9]

recursive_score_fns = [("Population", population_score),
                       ("Variance", variance_score),
                       ("Inertia", inertia_score),
                       ("Diameter", diameter_score),
                       ("MAD", mad_score),]

guided_score_fns = [("Error", error_score),
                    ("Impurity", impurity_score),
                    ("Label Entropy", label_entropy_score),
                    ("Margin", margin_score)]
km_interval = 0.1

trial_seeds = list(range(10))


