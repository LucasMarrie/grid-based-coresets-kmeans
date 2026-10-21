import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from benchmarking.experimental_constants import trial_seeds

def plot_vs_m(
    df: pd.DataFrame,
    test_name: str,
    method_names: list,
    k_value: int,
    baseline_method: str = "Identity",
    show_error_bars: bool = True,
    plot_total_time: bool = False,
    save_path: str = None,
):
    """
    Plots a bar chart of normalized Mean Cost or Total Time vs. m (Coreset Size / k) for each method,
    normalizing against a baseline method (e.g., Identity), with better axis scaling and m binning.

    Args:
        df: DataFrame containing the results.
        test_name: The test name to filter on.
        method_names: List of method names to include.
        k_value: The specific k value to filter by.
        baseline_method: Method to use for normalization.
        show_error_bars: Whether to include error bars.
        plot_total_time: If True, plots Total Time instead of Cost.
        save_path: If provided, saves the plot to this path.
        trial_seeds: List of seeds used to determine standard error.
    """
    df['Test Name'] = df['Test Name'].astype(str).str.strip()
    df = df[(df['Test Name'] == test_name) & (df['k'] == k_value)].copy()

    # Bin m values (modulo 10 rounding)
    df["m"] = (df["Coreset Size"] / df["k"]).round(-1).astype(int)

    # Get baseline values
    baseline_row = df[df['Method Name'] == baseline_method]
    if baseline_row.empty:
        raise ValueError(f"No baseline method '{baseline_method}' found for Test '{test_name}' and k={k_value}")

    if plot_total_time:
        baseline_val = (
            baseline_row["Mean Coreset Time (s)"].values[0] +
            baseline_row["Mean K-Means Time (s)"].values[0]
        )
    else:
        baseline_val = baseline_row["Mean Cost"].values[0]

    all_m = sorted(df[df['Method Name'] != baseline_method]["m"].unique())
    method_bar_data = {method: [] for method in method_names}
    method_bar_errs = {method: [] for method in method_names}

    for m in all_m:
        for method in method_names:
            method_df = df[(df['Method Name'] == method) & (df['m'] == m)]
            if not method_df.empty:
                if plot_total_time:
                    mean_val = method_df["Mean Coreset Time (s)"].values[0] + method_df["Mean K-Means Time (s)"].values[0]
                    std_val = np.sqrt(
                        method_df["Std Coreset Time (s)"].values[0]**2 +
                        method_df["Std K-Means Time (s)"].values[0]**2
                    )
                else:
                    mean_val = method_df["Mean Cost"].values[0]
                    std_val = method_df["Std Cost"].values[0]

                method_bar_data[method].append(mean_val / baseline_val)
                if show_error_bars:
                    method_bar_errs[method].append((std_val / baseline_val) / np.sqrt(len(trial_seeds)))
                else:
                    method_bar_errs[method].append(0)
            else:
                method_bar_data[method].append(np.nan)
                method_bar_errs[method].append(0)

    x = np.arange(len(all_m))
    total_width = 0.8
    bar_width = total_width / len(method_names)

    plt.figure(figsize=(12, 6))
    all_vals = []

    for i, method in enumerate(method_names):
        if method == baseline_method:
            continue
        y_vals = method_bar_data[method]
        y_errs = method_bar_errs[method]
        offset = (i - len(method_names) / 2) * bar_width + bar_width / 2
        plt.bar(x + offset,
                y_vals,
                bar_width,
                yerr=y_errs if show_error_bars else None,
                capsize=4,
                label=method)
        all_vals.extend([
            y + e for y, e in zip(y_vals, y_errs) if not np.isnan(y)
        ])
        all_vals.extend([
            y - e for y, e in zip(y_vals, y_errs) if not np.isnan(y)
        ])

    plt.xticks(x, [str(m) for m in all_m])
    plt.xlabel("m (Coreset Size / k)")
    ylabel = "Normalized Mean Total Time" if plot_total_time else "Normalized Mean Cost"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. m (Test: {test_name}, k={k_value}, normalized to '{baseline_method}')")
    plt.legend()
    plt.grid(True, axis='y')

    # Fit y-axis to visible data including error bars
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        if plot_total_time:
            plt.ylim(0, ymax + padding)
        else:
            plt.ylim(ymin - padding, ymax + padding)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_vs_m_avg_all(
    df: pd.DataFrame,
    method_names: list,
    baseline_method: str = "Identity",
    show_error_bars: bool = True,
    plot_total_time: bool = False,
    save_path: str = None,
):
    """
    Plots a bar chart of normalized average Mean Cost or Total Time vs. m across all datasets and k values.

    Args:
        df: DataFrame containing all experiment results.
        method_names: List of method names to include.
        baseline_method: Method to normalize against.
        show_error_bars: Whether to include standard error bars.
        plot_total_time: Whether to plot total time instead of cost.
        save_path: If provided, saves the plot to this path.
        trial_seeds: List of seeds used (used for standard error computation).
    """
    df['Test Name'] = df['Test Name'].astype(str).str.strip()
    df["m"] = (df["Coreset Size"] / df["k"]).round(-1).astype(int)

    # Normalize by baseline value per (Test Name, k) group
    norm_df = df.copy()
    norm_df["Normalized Value"] = np.nan
    norm_df["Normalized Std"] = np.nan

    for (test_name, k_val), group in df.groupby(["Test Name", "k"]):
        baseline_row = group[group["Method Name"] == baseline_method]
        if baseline_row.empty:
            continue

        if plot_total_time:
            baseline_val = (
                baseline_row["Mean Coreset Time (s)"].values[0] +
                baseline_row["Mean K-Means Time (s)"].values[0]
            )
        else:
            baseline_val = baseline_row["Mean Cost"].values[0]

        for idx, row in group.iterrows():
            if row["Method Name"] == baseline_method:
                continue
            if plot_total_time:
                val = row["Mean Coreset Time (s)"] + row["Mean K-Means Time (s)"]
                std = np.sqrt(row["Std Coreset Time (s)"]**2 + row["Std K-Means Time (s)"]**2)
            else:
                val = row["Mean Cost"]
                std = row["Std Cost"]
            norm_df.loc[idx, "Normalized Value"] = val / baseline_val
            norm_df.loc[idx, "Normalized Std"] = (std / baseline_val) / np.sqrt(len(trial_seeds))

    # Drop rows with NaNs in normalized values
    norm_df = norm_df.dropna(subset=["Normalized Value"])

    all_m = sorted(norm_df["m"].unique())
    method_bar_data = {method: [] for method in method_names}
    method_bar_errs = {method: [] for method in method_names}

    for m in all_m:
        for method in method_names:
            m_df = norm_df[(norm_df["Method Name"] == method) & (norm_df["m"] == m)]
            if not m_df.empty:
                mean_val = m_df["Normalized Value"].mean()
                std_err = m_df["Normalized Value"].std() / np.sqrt(len(m_df)) if show_error_bars else 0
                method_bar_data[method].append(mean_val)
                method_bar_errs[method].append(std_err)
            else:
                method_bar_data[method].append(np.nan)
                method_bar_errs[method].append(0)

    x = np.arange(len(all_m))
    total_width = 0.8
    bar_width = total_width / len(method_names)

    plt.figure(figsize=(12, 6))
    all_vals = []

    for i, method in enumerate(method_names):
        if method == baseline_method:
            continue
        y_vals = method_bar_data[method]
        y_errs = method_bar_errs[method]
        offset = (i - len(method_names) / 2) * bar_width + bar_width / 2
        plt.bar(x + offset,
                y_vals,
                bar_width,
                yerr=y_errs if show_error_bars else None,
                capsize=4,
                label=method)
        all_vals.extend([
            y + e for y, e in zip(y_vals, y_errs) if not np.isnan(y)
        ])
        all_vals.extend([
            y - e for y, e in zip(y_vals, y_errs) if not np.isnan(y)
        ])

    plt.xticks(x, [str(m) for m in all_m])
    plt.xlabel("m (Coreset Size / k)")
    ylabel = "Normalized Avg Total Time" if plot_total_time else "Normalized Avg Cost"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. m (averaged over all datasets and k, normalized to '{baseline_method}')")
    plt.legend()
    plt.grid(True, axis='y')

    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        padding = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        if plot_total_time:
            plt.ylim(0, ymax + padding)
        else:
            plt.ylim(ymin - padding, ymax + padding)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
