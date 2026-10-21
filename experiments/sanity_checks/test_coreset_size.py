from benchmarking.individual_tester import IndividualTestFramework
from benchmarking.datasets import get_test_datasets

from coresets.standard.identity import IdentityCoreset
from coresets.grid.grid_mean import GridMeanCoreset
from coresets.grid.search_grid_size import binary_search_grid_size

print("Preparing Tests and Datasets for coreset size...")
test_datasets = get_test_datasets(k_clusters=20)


target_cells_list = [1000, 2000, 5000, 10000]
tests = []
for test_dataset in test_datasets:
    for target_cells in target_cells_list:
        dataset = test_dataset['dataset']
        test = test_dataset.copy()
        method_name = f"Grid Mean"
        grid_size = binary_search_grid_size(dataset, target_cells)  
        method_object = GridMeanCoreset(grid_size=grid_size)
        test['method_name'] = method_name
        test['method_object'] = method_object
        tests.append(test)


trial_seeds = list(range(10))

test_framework = IndividualTestFramework(
    tests=tests,
    trial_seeds=trial_seeds
)

print("Running Coreset Size Sanity Tests...")
test_framework.run_tests()
test_framework.print_results()

import os
os.makedirs("sanity_check_results", exist_ok=True)
test_framework.save_results("sanity_check_results/coreset_size.csv")