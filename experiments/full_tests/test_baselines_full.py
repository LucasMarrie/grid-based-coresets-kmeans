"""
Full tests for baseline methods
"""

from benchmarking.individual_tester import IndividualTestFramework

from coresets.standard.identity import IdentityCoreset
from coresets.standard.random import RandomSamplingCoreset
from coresets.grid.grid_mean import GridMeanCoreset

from coresets.grid.search_grid_size import binary_search_grid_size

from benchmarking.experimental_constants import (
    test_datasets,
    ms,
    trial_seeds,
)
import os



tests = []
for test_dataset in test_datasets:
    tests.append(
        {**test_dataset.copy(), "method_name": "Identity", "method_object": IdentityCoreset()}
    )
    for m in ms:
        target_cells = m * test_dataset["k_clusters"]
        dataset = test_dataset["dataset"]

        grid_size = binary_search_grid_size(dataset, target_cells)

        random_coreset = RandomSamplingCoreset(target_cells)
        grid_mean_coreset = GridMeanCoreset(grid_size=grid_size)
        
        tests += [
            {**test_dataset.copy(),
             "method_name": "Random Sampling",
             "method_object": random_coreset},
            {**test_dataset.copy(),
             "method_name": "Grid Mean",
             "method_object": grid_mean_coreset},
        ]



test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Baseline Full test...")
test_framework.run_tests(verbose=True)
test_framework.print_results()
# create results directory 
os.makedirs("results", exist_ok=True)
test_framework.save_results("results/baseline_full.csv")
