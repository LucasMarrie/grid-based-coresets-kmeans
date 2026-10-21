"""
Full test for recursive method
"""

from benchmarking.individual_tester import IndividualTestFramework

from coresets.grid.recursive import RecursiveGridCoreset

from coresets.grid.search_grid_size import binary_search_grid_size

from benchmarking.experimental_constants import (
    get_full_test_datasets,
    ms,
    recursive_score_fns,
    starting_factors,
    trial_seeds,
)
import os




test_datasets = get_full_test_datasets()

tests = []
for test_dataset in test_datasets:
    for m in ms:
        target_cells = m * test_dataset["k_clusters"]
        dataset = test_dataset["dataset"]

        for starting_factor in starting_factors:
            starting_grid_size = binary_search_grid_size(
                dataset, target_cells * starting_factor
            )
            for score_name, score_fn in recursive_score_fns:
                recursive_grid_coreset = RecursiveGridCoreset(
                    starting_grid_size=starting_grid_size,
                    target_coreset_size=target_cells,
                    score_fn=score_fn,
                )
                tests.append({
                    **test_dataset.copy(),
                    "method_name": f"Recursive ({score_name}) s={starting_factor}",
                    "method_object": recursive_grid_coreset,
                })
  




test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Recursive Full test...")
test_framework.run_tests(verbose=True)
test_framework.print_results()
# create results directory 
os.makedirs("results", exist_ok=True)
test_framework.save_results("results/test_recursive_full.csv")
