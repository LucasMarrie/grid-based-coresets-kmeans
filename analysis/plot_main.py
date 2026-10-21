"""
Generate every plot in plots/ from the full test results in results/.

Run from the repository root:
    python analysis/plot_main.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # plots are only saved to disk, never shown
import pandas as pd
from plot_against_m import plot_vs_m, plot_vs_m_avg_all
from plot_against_k import plot_vs_k, plot_vs_k_avg_all

# coreset multiplier used for the plots against k
CHOSEN_M = 100


def main():
    baseline_df = pd.read_csv("results/baseline_full.csv")
    recursive_df = pd.read_csv("results/test_recursive_full.csv")
    guided_df = pd.read_csv("results/test_guided_full.csv")

    # merge all dataframes
    merged_df = pd.concat([baseline_df, recursive_df, guided_df], ignore_index=True)

    tests = merged_df["Test Name"].unique().tolist()
    ks = recursive_df["k"].unique().tolist()

    recursive_methods = recursive_df["Method Name"].unique().tolist()
    guided_methods = guided_df["Method Name"].unique().tolist()

    final_methods = [
        "Identity",
        "Random Sampling",
        "Grid Mean",
        "Recursive (Inertia) s=0.1",
        "Guided (Error) s=0.9",
    ]

    # (output folder, methods to compare, whether to also plot total time)
    plot_groups = [
        # score function comparison at the coarse start factor
        ("plots/recursive", [name for name in recursive_methods if "s=0.1" in name], True),
        ("plots/guided", [name for name in guided_methods if "s=0.1" in name], True),
        # start factor comparison for the best score function
        ("plots/recursive_start", ["Recursive (Inertia) s=0.1", "Recursive (Inertia) s=0.9"], True),
        ("plots/guided_start", ["Guided (Error) s=0.1", "Guided (Error) s=0.9"], True),
        # final comparison of all methods
        ("plots/final", final_methods, True),
        ("plots/final_no_random", [name for name in final_methods if name != "Random Sampling"], False),
    ]

    for folder, methods, include_time in plot_groups:
        print(f"Generating plots in {folder}...")
        metrics = [("cost", False), ("time", True)] if include_time else [("cost", False)]

        for metric, plot_total_time in metrics:
            m_dir = f"{folder}/{metric}_against_m"
            k_dir = f"{folder}/{metric}_against_k"
            os.makedirs(m_dir, exist_ok=True)
            os.makedirs(k_dir, exist_ok=True)

            # against m
            plot_vs_m_avg_all(merged_df, methods, plot_total_time=plot_total_time,
                              save_path=f"{m_dir}/all_avg.png")
            for test in tests:
                for k in ks:
                    plot_vs_m(merged_df, test, methods, k, plot_total_time=plot_total_time,
                              save_path=f"{m_dir}/{test}_k={k}.png")

            # against k
            plot_vs_k_avg_all(merged_df, methods, CHOSEN_M, plot_total_time=plot_total_time,
                              save_path=f"{k_dir}/all_avg_m={CHOSEN_M}.png")
            for test in tests:
                plot_vs_k(merged_df, test, methods, CHOSEN_M, plot_total_time=plot_total_time,
                          save_path=f"{k_dir}/{test}_m={CHOSEN_M}.png")


if __name__ == "__main__":
    main()
