"""
Experimental constants for the full benchmarking suite. (Not for preliminary tests)

Datasets are built lazily through get_full_test_datasets() so that importing this module
(e.g. from the analysis scripts) does not trigger any data downloads or generation.
"""

from benchmarking.datasets import get_test_datasets
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

ms = [50, 100, 200, 500]
ks = [10, 20, 30, 40, 50]


def get_full_test_datasets():
    test_datasets = []
    for k in ks:
        test_datasets += get_test_datasets(k)
    return test_datasets


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
