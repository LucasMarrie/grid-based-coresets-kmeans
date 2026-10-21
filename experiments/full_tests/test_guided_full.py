"""
Full test for guided method
"""

from benchmarking.individual_tester import IndividualTestFramework

from coresets.grid.guided import KMeansGuidedRecursiveGrid

from coresets.grid.search_grid_size import binary_search_grid_size

from benchmarking.experimental_constants import (
    get_full_test_datasets,
    ms,
    guided_score_fns,
    starting_factors,
    km_interval,
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

            for score_name, score_fn in guided_score_fns:
                kmeans_guided_recursive_grid = KMeansGuidedRecursiveGrid(
                    starting_grid_size=starting_grid_size,
                    target_coreset_size=target_cells,
                    score_fn=score_fn,
                    km_interval=km_interval,
                )
                tests.append({
                    **test_dataset.copy(),
                    "method_name": f"Guided ({score_name}) s={starting_factor}",
                    "method_object": kmeans_guided_recursive_grid,
                })            




test_framework = IndividualTestFramework(tests=tests, trial_seeds=trial_seeds)

print("Running Guided Full test...")
test_framework.run_tests(verbose=True)
test_framework.print_results()
# create results directory 
os.makedirs("results", exist_ok=True)
test_framework.save_results("results/test_guided_full.csv")
