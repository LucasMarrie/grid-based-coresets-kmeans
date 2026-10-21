import pandas as pd
from plot_against_m import plot_vs_m, plot_vs_m_avg_all
from plot_against_k import plot_vs_k, plot_vs_k_avg_all

baseline_df = pd.read_csv("results/baseline_full.csv")
recursive_df = pd.read_csv("results/test_recursive_full.csv")
guided_df = pd.read_csv("results/test_guided_full.csv")

# merge all dataframes
merged_df = pd.concat([baseline_df, recursive_df, guided_df], ignore_index=True)

tests = merged_df["Test Name"].unique().tolist()

recursive_methods = recursive_df.get("Method Name").unique().tolist()
guided_methods = guided_df.get("Method Name").unique().tolist()

ks = recursive_df["k"].unique().tolist()

recursive_start_01 = [name for name in recursive_methods if "s=0.1" in name]

recursive_detail = [
    "Recursive (Inertia) s=0.1",
    "Recursive (Inertia) s=0.9",
]

guided_start_01 = [name for name in guided_methods if "s=0.1" in name]

guided_detail = [
    "Guided (Error) s=0.1",
    "Guided (Error) s=0.9",
]

final_methods = [
    "Identity",
    "Random Sampling",
    "Grid Mean",
    "Recursive (Inertia) s=0.1",
    "Guided (Error) s=0.9",
]

final_methods_no_random = [
    "Identity",
    "Grid Mean",
    "Recursive (Inertia) s=0.1",
    "Guided (Error) s=0.9",
]

folders = [ "plots/recursive/cost_against_m",
            "plots/recursive/time_against_m",
            "plots/guided/cost_against_m",
            "plots/guided/time_against_m",
            "plots/recursive/cost_against_k",
            "plots/recursive/time_against_k",
            "plots/guided/cost_against_k",
            "plots/guided/time_against_k",
            "plots/final/cost_against_k",
            "plots/final/time_against_k",
            "plots/final/cost_against_m",
            "plots/final/time_against_m",
            "plots/recursive_start/cost_against_m",
            "plots/recursive_start/time_against_m",
            "plots/guided_start/cost_against_m",
            "plots/guided_start/time_against_m",
            "plots/recursive_start/cost_against_k",
            "plots/recursive_start/time_against_k",
            "plots/guided_start/cost_against_k",
            "plots/guided_start/time_against_k",
            "plots/final_no_random/cost_against_m",
            "plots/final_no_random/cost_against_k",]

# Create directories if they do not exist
for folder in folders:
    import os
    os.makedirs(folder, exist_ok=True)

#recursive
plot_vs_m_avg_all(merged_df, recursive_start_01, save_path=f"plots/recursive/cost_against_m/all_avg.png")
plot_vs_m_avg_all(merged_df, recursive_start_01, plot_total_time=True, save_path=f"plots/recursive/time_against_m/all_avg.png")

plot_vs_m_avg_all(merged_df, recursive_detail, save_path=f"plots/recursive_start/cost_against_m/all_avg.png")
plot_vs_m_avg_all(merged_df, recursive_detail, plot_total_time=True, save_path=f"plots/recursive_start/time_against_m/all_avg.png")


#guided
plot_vs_m_avg_all(merged_df, guided_start_01, save_path=f"plots/guided/cost_against_m/all_avg.png")
plot_vs_m_avg_all(merged_df, guided_start_01, plot_total_time=True, save_path=f"plots/guided/time_against_m/all_avg.png")

plot_vs_m_avg_all(merged_df, guided_detail, save_path=f"plots/guided_start/cost_against_m/all_avg.png")
plot_vs_m_avg_all(merged_df, guided_detail, plot_total_time=True, save_path=f"plots/guided_start/time_against_m/all_avg.png")

#final 
plot_vs_m_avg_all(merged_df, final_methods, save_path=f"plots/final/cost_against_m/all_avg.png")
plot_vs_m_avg_all(merged_df, final_methods_no_random, save_path=f"plots/final_no_random/cost_against_m/all_avg.png")

plot_vs_m_avg_all(merged_df, final_methods, plot_total_time=True, save_path=f"plots/final/time_against_m/all_avg.png")

# Perform M plots
for test in tests:
    for k in ks:
        #recursive
        plot_vs_m(merged_df, test, recursive_start_01, k, save_path=f"plots/recursive/cost_against_m/{test}_k={k}.png")
        plot_vs_m(merged_df, test, recursive_start_01, k, plot_total_time=True, save_path=f"plots/recursive/time_against_m/{test}_k={k}.png")

        plot_vs_m(merged_df, test, recursive_detail, k, save_path=f"plots/recursive_start/cost_against_m/{test}_k={k}.png")
        plot_vs_m(merged_df, test, recursive_detail, k, plot_total_time=True, save_path=f"plots/recursive_start/time_against_m/{test}_k={k}.png")

        #guided
        plot_vs_m(merged_df, test, guided_start_01, k, save_path=f"plots/guided/cost_against_m/{test}_k={k}.png")
        plot_vs_m(merged_df, test, guided_start_01, k, plot_total_time=True, save_path=f"plots/guided/time_against_m/{test}_k={k}.png")

        plot_vs_m(merged_df, test, guided_detail, k, save_path=f"plots/guided_start/cost_against_m/{test}_k={k}.png")
        plot_vs_m(merged_df, test, guided_detail, k, plot_total_time=True, save_path=f"plots/guided_start/time_against_m/{test}_k={k}.png")

        # Final
        plot_vs_m(merged_df, test, final_methods, k, save_path=f"plots/final/cost_against_m/{test}_k={k}.png")
        plot_vs_m(merged_df, test, final_methods_no_random, k, save_path=f"plots/final_no_random/cost_against_m/{test}_k={k}.png")

        plot_vs_m(merged_df, test, final_methods, k, plot_total_time=True, save_path=f"plots/final/time_against_m/{test}_k={k}.png")


chosen_m = 100

#recursive
plot_vs_k_avg_all(merged_df, recursive_start_01, chosen_m, save_path=f"plots/recursive/cost_against_k/all_avg_m={chosen_m}.png")
plot_vs_k_avg_all(merged_df, recursive_start_01, chosen_m, plot_total_time=True, save_path=f"plots/recursive/time_against_k/all_avg_m={chosen_m}.png")

plot_vs_k_avg_all(merged_df, recursive_detail, chosen_m, save_path=f"plots/recursive_start/cost_against_k/all_avg_m={chosen_m}.png")
plot_vs_k_avg_all(merged_df, recursive_detail, chosen_m, plot_total_time=True, save_path=f"plots/recursive_start/time_against_k/all_avg_m={chosen_m}.png")

#guided
plot_vs_k_avg_all(merged_df, guided_start_01, chosen_m, save_path=f"plots/guided/cost_against_k/all_avg_m={chosen_m}.png")
plot_vs_k_avg_all(merged_df, guided_start_01, chosen_m, plot_total_time=True, save_path=f"plots/guided/time_against_k/all_avg_m={chosen_m}.png")

plot_vs_k_avg_all(merged_df, guided_detail, chosen_m, save_path=f"plots/guided_start/cost_against_k/all_avg_m={chosen_m}.png")
plot_vs_k_avg_all(merged_df, guided_detail, chosen_m, plot_total_time=True, save_path=f"plots/guided_start/time_against_k/all_avg_m={chosen_m}.png")

#final 
plot_vs_k_avg_all(merged_df, final_methods, chosen_m, save_path=f"plots/final/cost_against_k/all_avg_m={chosen_m}.png")
plot_vs_k_avg_all(merged_df, final_methods_no_random, chosen_m, save_path=f"plots/final_no_random/cost_against_k/all_avg_m={chosen_m}.png")

plot_vs_k_avg_all(merged_df, final_methods, chosen_m, plot_total_time=True, save_path=f"plots/final/time_against_k/all_avg_m={chosen_m}.png")

for test in tests:
    #recursive
    plot_vs_k(merged_df, test, recursive_start_01, chosen_m, save_path=f"plots/recursive/cost_against_k/{test}_m={chosen_m}.png")
    plot_vs_k(merged_df, test, recursive_start_01, chosen_m, plot_total_time=True, save_path=f"plots/recursive/time_against_k/{test}_m={chosen_m}.png")

    plot_vs_k(merged_df, test, recursive_detail, chosen_m, save_path=f"plots/recursive_start/cost_against_k/{test}_m={chosen_m}.png")
    plot_vs_k(merged_df, test, recursive_detail, chosen_m, plot_total_time=True, save_path=f"plots/recursive_start/time_against_k/{test}_m={chosen_m}.png")

    #guided
    plot_vs_k(merged_df, test, guided_start_01, chosen_m, save_path=f"plots/guided/cost_against_k/{test}_m={chosen_m}.png")
    plot_vs_k(merged_df, test, guided_start_01, chosen_m, plot_total_time=True, save_path=f"plots/guided/time_against_k/{test}_m={chosen_m}.png")

    plot_vs_k(merged_df, test, guided_detail, chosen_m, save_path=f"plots/guided_start/cost_against_k/{test}_m={chosen_m}.png")
    plot_vs_k(merged_df, test, guided_detail, chosen_m, plot_total_time=True, save_path=f"plots/guided_start/time_against_k/{test}_m={chosen_m}.png")

    # Final 
    plot_vs_k(merged_df, test, final_methods, chosen_m, save_path=f"plots/final/cost_against_k/{test}_m={chosen_m}.png")
    plot_vs_k(merged_df, test, final_methods_no_random, chosen_m, save_path=f"plots/final_no_random/cost_against_k/{test}_m={chosen_m}.png")

    plot_vs_k(merged_df, test, final_methods, chosen_m, plot_total_time=True, save_path=f"plots/final/time_against_k/{test}_m={chosen_m}.png")

