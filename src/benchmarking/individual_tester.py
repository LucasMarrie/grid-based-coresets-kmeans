import numpy as np
import time
import statistics
from sklearn.cluster import KMeans

class IndividualTestFramework():


    def __init__(self, tests, trial_seeds, n_init=10):
        """
        Initialize the testing framework.

        Parameters:
          - test: A list of dictionaries, each containing:
              'name': name of the dataset,
              'dataset': a NumPy array representing the dataset,
              'k_clusters': number of clusters for k-means.
              'method_name': The name of the coreset method.
              'method_object': The coreset method, an instance of AbstractCoreset with the following signature:
                - get_coreset(data, k_clusters) -> coreset
          - trial_seeds: A list of random seeds for reproducibility to use for each trial, len(trial_seeds) is the number of trials ran.
        """
        self.tests = tests
        self.trial_seeds = trial_seeds
        self.n_init = n_init
        self.results = {}  


    def compute_cost(self, full_data, centers):
        """
        Compute the cost of clustering using the given centers on the full dataset.
        """
        distances = np.linalg.norm(full_data[:, np.newaxis] - centers, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

    def run_single_trial(self, data, k_clusters, coreset_obj, seed):
        """
        Run a single trial for a given test and coreset method.
        """

        # Measure coreset generation time.
        start = time.time()
        coreset_result = coreset_obj.get_coreset(data, k_clusters, seed)
        coreset_time = time.time() - start

        # If the coreset function returns a tuple (coreset, sample_weights) then unpack.
        if isinstance(coreset_result, tuple):
            coreset_data, sample_weights = coreset_result
        else:
            coreset_data = coreset_result
            sample_weights = None

        # Run k-means on the coreset.
        start = time.time()
        # if kmeans init is not a string (meaning it is a numpy array), set n_init to 1.
        n_init = 1 if type(coreset_obj.kmeans_init) is not str else self.n_init
        kmeans = KMeans(n_clusters=k_clusters, random_state=seed, init=coreset_obj.kmeans_init, n_init=n_init)
        if sample_weights is not None:
            kmeans.fit(coreset_data, sample_weight=sample_weights)
        else:
            kmeans.fit(coreset_data)
        kmeans_time = time.time() - start
        
        centers = kmeans.cluster_centers_

        cost = self.compute_cost(data, centers)
        return len(coreset_data), coreset_time, kmeans_time, cost

    def run_tests(self, verbose=False):
        for test in self.tests:
            test_name = test.get("name", "Unnamed Test")
            data = test["dataset"]
            k_clusters = test["k_clusters"]
            method_name = test["method_name"]
            method_obj = test["method_object"]

            if test_name not in self.results:
                self.results[test_name] = {}
            if k_clusters not in self.results[test_name]:
                self.results[test_name][k_clusters] = {}
            if method_name not in self.results[test_name][k_clusters]:
                self.results[test_name][k_clusters][method_name] = []

            if verbose:
                print(f"Running test: {test_name} on {method_name}")

            coreset_sizes = []
            coreset_times = []
            kmeans_times = []
            costs = []

            for seed in self.trial_seeds:
                cs, ct, kt, cost = self.run_single_trial(
                    data, k_clusters, method_obj, seed
                )
                coreset_sizes.append(cs)
                coreset_times.append(ct)
                kmeans_times.append(kt)
                costs.append(cost)

            median_coreset_time = statistics.median(coreset_times)
            median_kmeans_time = statistics.median(kmeans_times)
            median_error = statistics.median(costs)
            median_coreset_size = statistics.median(coreset_sizes)  

            # store mean and standard deviation of the results
            mean_coreset_time = statistics.mean(coreset_times)
            mean_kmeans_time = statistics.mean(kmeans_times)
            mean_error = statistics.mean(costs)
            std_coreset_time = statistics.stdev(coreset_times) if len(coreset_times) > 1 else 0
            std_kmeans_time = statistics.stdev(kmeans_times) if len(kmeans_times) > 1 else 0
            std_error = statistics.stdev(costs) if len(costs) > 1 else 0

            self.results[test_name][k_clusters][method_name].append({
                "median_coreset_size": median_coreset_size,
                "median_coreset_time": median_coreset_time,
                "median_kmeans_time": median_kmeans_time,
                "median_cost": median_error,
                "mean_coreset_time": mean_coreset_time,
                "mean_kmeans_time": mean_kmeans_time,
                "mean_cost": mean_error,
                "std_coreset_time": std_coreset_time,
                "std_kmeans_time": std_kmeans_time,
                "std_cost": std_error
            })


    def print_results(self):
        """
        Print the test results in a formatted table.
        """
        for test_name, k_clusters in self.results.items():
            for k, methods in k_clusters.items():
                print(f"\nResults for Test: {test_name}, k = {k}")
                print("Method Name                                Corset Size   Coreset Time (s)   K-Means Time (s)              Cost")
                print("-" * (35+19+19+19+21+19))
                for method_name, metrics_list in methods.items():
                    for metrics in metrics_list:
                        print(f"{method_name:35s} {metrics['median_coreset_size']:18.0f} {metrics['median_coreset_time']:18.4f} "
                            f"{metrics['median_kmeans_time']:18.4f} {metrics['median_cost']:18.4f}")

    def save_results(self, filename):
        """
        Save the results to a CSV file.
        """
        import csv
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Test Name', 'k', 'Method Name', 'Coreset Size', 'Coreset Time (s)', 'K-Means Time (s)', 'Cost', 'Mean Coreset Time (s)', 'Std Coreset Time (s)', 'Mean K-Means Time (s)', 'Std K-Means Time (s)', 'Mean Cost', 'Std Cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for test_name, k_clusters in self.results.items():
                for k, methods in k_clusters.items():
                    for method_name, metrics_list in methods.items():
                        for metrics in metrics_list:
                            writer.writerow({
                                'Test Name': test_name,
                                'k': k,
                                'Method Name': method_name,
                                'Coreset Size': metrics['median_coreset_size'],
                                'Coreset Time (s)': metrics['median_coreset_time'],
                                'K-Means Time (s)': metrics['median_kmeans_time'],
                                'Cost': metrics['median_cost'],
                                'Mean Coreset Time (s)': metrics['mean_coreset_time'],
                                'Std Coreset Time (s)': metrics['std_coreset_time'],
                                'Mean K-Means Time (s)': metrics['mean_kmeans_time'],
                                'Std K-Means Time (s)': metrics['std_kmeans_time'],
                                'Mean Cost': metrics['mean_cost'],
                                'Std Cost': metrics['std_cost']
                            })

