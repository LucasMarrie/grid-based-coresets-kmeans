"""
Preliminary tests for guided start factor on default settings
"""

from benchmarking.individual_tester import IndividualTestFramework
from benchmarking.datasets import (
    get_blobs_2d,
    get_blobs_3d,
    get_covertype_data,
    get_us_census_data,
)

from coresets.standard.identity import IdentityCoreset
from coresets.grid.guided import KMeansGuidedRecursiveGrid
from coresets.grid.score_heuristics import (
    error_score,
)
from coresets.grid.search_grid_size import binary_search_grid_size

import os

print("Preparing Tests and Datasets for guided start factor...")
covertype_2d, covertype_3d = get_covertype_data()
us_census_2d, us_census_3d = get_us_census_data()

test_datasets = [
    {"name": "2d blobs", "dataset": get_blobs_2d(1000000, 20), "k_clusters": 20},
    {"name": "3d blobs", "dataset": get_blobs_3d(1000000, 20), "k_clusters": 20},
    {"name": "Covertype 2d", "dataset": covertype_2d, "k_clusters": 20},
    {"name": "Covertype 3d", "dataset": covertype_3d, "k_clusters": 20},
    {"name": "US Census 2d", "dataset": us_census_2d, "k_clusters": 20},
    {"name": "US Census 3d", "dataset": us_census_3d, "k_clusters": 20},
]


starting_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
target_cells_list = [10000] #[1000, 2000, 5000, 10000]
tests = []
for test_dataset in test_datasets:
    # add identity coreset test
    tests.append(
        {**test_dataset.copy(), "method_name": "Identity", "method_object": IdentityCoreset()
        })
    for target_cells in target_cells_list:
        dataset = test_dataset["dataset"]
        for starting_factor in starting_factors:
            starting_grid_size = binary_search_grid_size(
                dataset, target_cells * starting_factor
            )
            score_fns = {
                'error': error_score,
            }
            for score_name, score_fn in score_fns.items():
                tests.append(
                    {
                        **test_dataset.copy(),
                        "method_name": f"Guided ({score_name}) s={starting_factor}",
                        "method_object": KMeansGuidedRecursiveGrid(
                            starting_grid_size=starting_grid_size, target_coreset_size=target_cells, score_fn=score_fn
                        ),
                    },
                )

trial_seeds = list(range(10))

test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Guided start factor tests...")
test_framework.run_tests(verbose=True)
test_framework.print_results()
# create results directory
os.makedirs("results", exist_ok=True)
test_framework.save_results("results/test_guided_start_factor.csv")
