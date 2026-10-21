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

print("Preparing Tests and Datasets for guided...")
covertype_2d, covertype_3d = get_covertype_data()
us_census_2d, us_census_3d = get_us_census_data()

test_datasets = [
    {"name": "2d blobs", "dataset": get_blobs_2d(1000000, 20), "k_clusters": 20},
    {"name": "3d blobs", "dataset": get_blobs_3d(1000000, 20), "k_clusters": 20},
]


starting_factors = [0.3] 
target_cells_list = [10000] 
tests = []
for test_dataset in test_datasets:
    tests.append(
        {**test_dataset.copy(), "method_name": "Identity", "method_object": IdentityCoreset()}
    )
    for target_cells in target_cells_list:
        dataset = test_dataset["dataset"]
        for starting_factor in starting_factors:
            starting_grid_size = binary_search_grid_size(
                dataset, target_cells * starting_factor
            )
            tests.append(
                {
                    **test_dataset.copy(),
                    "method_name": f"Guided default s={starting_factor}",
                    "method_object": KMeansGuidedRecursiveGrid(
                        starting_grid_size=starting_grid_size,
                        target_coreset_size=target_cells,
                        score_fn=error_score,
                    ),
                }
            )
            km_intervals = [0.01,0.05, 0.1, 1]
            max_iters = [2, 20, 100]
            for km_interval in km_intervals:
                for max_iter in max_iters:
                    tests.append(
                        {
                            **test_dataset.copy(),
                            "method_name": f"Guided kmi={km_interval} mxi={max_iter}  s={starting_factor}",
                            "method_object": KMeansGuidedRecursiveGrid(
                                starting_grid_size=starting_grid_size,
                                target_coreset_size=target_cells,
                                score_fn=error_score,
                                km_interval=km_interval,
                                km_kwargs={"max_iter": max_iter},
                            ),
                        }
                    )


trial_seeds = list(range(10))

test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Guided Sanity Tests...")
test_framework.run_tests(verbose=True)
test_framework.print_results()

import os
os.makedirs("sanity_check_results", exist_ok=True)
test_framework.save_results("sanity_check_results/guided.csv")