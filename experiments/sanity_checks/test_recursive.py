from benchmarking.individual_tester import IndividualTestFramework
from benchmarking.datasets import get_test_datasets

from coresets.grid.recursive import RecursiveGridCoreset
from coresets.standard.identity import IdentityCoreset
from coresets.grid.score_heuristics import (
    population_score,
)
from coresets.grid.grid_mean import GridMeanCoreset
from coresets.grid.search_grid_size import binary_search_grid_size

print("Preparing Tests and Datasets for recursive...")
test_datasets = get_test_datasets(k_clusters=20, names=["2d blobs", "3d blobs"])


starting_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
target_cells_list = [10000] 
tests = []
for test_dataset in test_datasets:
    tests.append(
        {**test_dataset.copy(), "method_name": "Identity", "method_object": IdentityCoreset()}
    )
    for target_cells in target_cells_list:
        dataset = test_dataset["dataset"]
        tests.append(
            {
                **test_dataset.copy(),
                "method_name": f"Grid Mean",
                "method_object": GridMeanCoreset(grid_size=binary_search_grid_size(
                dataset, target_cells
            ))
            }
        )
        for starting_factor in starting_factors:
            starting_grid_size = binary_search_grid_size(
                dataset, target_cells * starting_factor
            )
            score_fns = {
                'population': population_score,
            }
            for score_name, score_fn in score_fns.items():
                tests.append(
                    {
                        **test_dataset.copy(),
                        "method_name": f"Recursive ({score_name}) s={starting_factor}",
                        "method_object": RecursiveGridCoreset(
                            starting_grid_size=starting_grid_size, target_coreset_size=target_cells, score_fn=score_fn
                        ),
                    }
                )

trial_seeds = list(range(10))

test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Recursive Sanity Tests...")
test_framework.run_tests(verbose=True)
test_framework.print_results()

import os
os.makedirs("sanity_check_results", exist_ok=True)
test_framework.save_results("sanity_check_results/recursive.csv")