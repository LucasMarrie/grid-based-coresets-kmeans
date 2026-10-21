"""
Preliminary tests for guided functions with re-use centers enabled
"""

from benchmarking.individual_tester import IndividualTestFramework
from benchmarking.datasets import get_test_datasets

from coresets.standard.identity import IdentityCoreset
from coresets.grid.guided import KMeansGuidedRecursiveGrid
from coresets.grid.score_heuristics import (
    impurity_score,
    error_score,
    margin_score,
    label_entropy_score
)
from coresets.grid.search_grid_size import binary_search_grid_size

import os

print("Preparing Tests and Datasets for guided fns...")
test_datasets = get_test_datasets(k_clusters=20)

starting_factors = [0.3] #[0.1, 0.3, 0.5, 0.7, 0.9]
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
                'impurity': impurity_score,
                'error': error_score,
                'margin': margin_score,
                'label_entropy': label_entropy_score
            }
            for score_name, score_fn in score_fns.items():
                tests.append(
                    {
                        **test_dataset.copy(),
                        "method_name": f"Guided ({score_name}) s={starting_factor}",
                        "method_object": KMeansGuidedRecursiveGrid(
                            starting_grid_size=starting_grid_size, target_coreset_size=target_cells, score_fn=score_fn, km_reuse_centers=True
                        ),
                    }
                )

trial_seeds = list(range(10))

test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Guided Fns test...")
test_framework.run_tests(verbose=True)
test_framework.print_results()
# create results directory
os.makedirs("results", exist_ok=True)
test_framework.save_results("results/test_guided_fns_reuse.csv")
