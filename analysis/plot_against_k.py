import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from benchmarking.experimental_constants import trial_seeds


def plot_vs_k(
    df: pd.DataFrame,
    test_name: str,
    method_names: list,
    m_value: int,
    baseline_method: str = "Identity",
    show_error_bars: bool = True,
    plot_total_time: bool = False,
    save_path: str = None,
):
    """
    Plots normalized Mean Cost or Total Time vs. k with grouped bars, normalized to a baseline method.
    Adjusts y-axis limits to include error bars.
    """

    def update_yaxis_with_errors(grouped_data):
        """Compute ymin and ymax including error bars."""
        all_vals = []
        for vals, errs in grouped_data.values():
            for val, err in zip(vals, errs):
                if not np.isnan(val):
                    all_vals.extend([val + err, val - err])
        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals)
            padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
            return y_min - padding, y_max + padding
        return None, None

    df['Test Name'] = df['Test Name'].astype(str).str.strip()
    df = df[df['Test Name'] == test_name]

    grouped_data = {}
    ks = sorted(df['k'].unique())

    # Compute baseline values
    baseline_values = {}
    for k in ks:
        target_size = k * m_value
        threshold = 0.05 * target_size
        baseline_row = df[
            (df['Method Name'] == baseline_method) &
            (df['k'] == k)
        ]
        if not baseline_row.empty:
            if plot_total_time:
                baseline_val = (
                    baseline_row["Mean Coreset Time (s)"].values[0] +
                    baseline_row["Mean K-Means Time (s)"].values[0]
                )
            else:
                baseline_val = baseline_row["Mean Cost"].values[0]
        else:
            baseline_val = np.nan
        baseline_values[k] = baseline_val

    # Gather and normalize data
    for method in method_names:
        method_data = []
        method_error = []
        for k in ks:
            target_size = k * m_value
            threshold = 0.05 * target_size
            row = df[
                (df['Method Name'] == method) &
                (df['k'] == k) &
                ((df['Coreset Size'].between(target_size - threshold, target_size + threshold))
                 | (method == baseline_method))
            ]
            if not row.empty and not np.isnan(baseline_values[k]):
                if plot_total_time:
                    mean = row["Mean Coreset Time (s)"].values[0] + row["Mean K-Means Time (s)"].values[0]
                    if show_error_bars:
                        std = np.sqrt(
                            row["Std Coreset Time (s)"].values[0]**2 +
                            row["Std K-Means Time (s)"].values[0]**2
                        )
                    else:
                        std = 0
                else:
                    mean = row["Mean Cost"].values[0]
                    std = row["Std Cost"].values[0] if show_error_bars else 0

                normalized_mean = mean / baseline_values[k]
                normalized_se = std / baseline_values[k] / np.sqrt(len(trial_seeds)) if show_error_bars else 0
            else:
                normalized_mean = np.nan
                normalized_se = 0

            method_data.append(normalized_mean)
            method_error.append(normalized_se)
        grouped_data[method] = (method_data, method_error)

    # Plotting
    x = np.arange(len(ks))  # label locations
    total_width = 0.8
    bar_width = total_width / len(method_names)

    plt.figure(figsize=(12, 6))
    for i, method in enumerate(method_names):
        data, errors = grouped_data[method]
        offset = (i - len(method_names) / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, data, bar_width,
                yerr=errors if show_error_bars else None,
                capsize=4, label=method)

    # Adjust y-axis to zoom into the actual data range (including error bars)
    y_min, y_max = update_yaxis_with_errors(grouped_data)
    if plot_total_time:
        y_min = 0
    if y_min is not None:
        plt.ylim(y_min, y_max)

    plt.xticks(x, ks)
    plt.xlabel("k (Number of Clusters)")
    ylabel = "Normalized Mean Total Time" if plot_total_time else "Normalized Mean Cost"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. k (Test: {test_name}, m = {m_value}, normalized to '{baseline_method}')")
    plt.legend()
    plt.grid(True, axis='y')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_vs_k_avg_all(
    df: pd.DataFrame,
    method_names: list,
    m_value: int,
    baseline_method: str = "Identity",
    show_error_bars: bool = True,
    plot_total_time: bool = False,
    save_path: str = None,
):
    """
    Plots normalized average Mean Cost or Total Time vs. k, averaged across all datasets,
    normalized to the baseline method per k.

    Args:
        df: DataFrame containing all experiment results.
        method_names: List of method names to include.
        m_value: The m value used to calculate target coreset size.
        baseline_method: Method to normalize against.
        show_error_bars: Whether to include standard error bars.
        plot_total_time: Whether to plot total time instead of cost.
        save_path: If provided, saves the plot to this path.
        trial_seeds: List of seeds used (used for standard error computation).
    """
    df['Test Name'] = df['Test Name'].astype(str).str.strip()

    grouped_data = {}
    ks = sorted(df['k'].unique())

    # Normalize by baseline per (Test Name, k)
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

    for method in method_names:
        method_means = []
        method_errors = []
        for k in ks:
            method_k_df = norm_df[(norm_df["Method Name"] == method) & (norm_df["k"] == k)]
            if not method_k_df.empty:
                mean_val = method_k_df["Normalized Value"].mean()
                std_err = method_k_df["Normalized Value"].std() / np.sqrt(len(method_k_df)) if show_error_bars else 0
                method_means.append(mean_val)
                method_errors.append(std_err)
            else:
                method_means.append(np.nan)
                method_errors.append(0)
        grouped_data[method] = (method_means, method_errors)

    # Plotting
    x = np.arange(len(ks))  # label locations
    total_width = 0.8
    bar_width = total_width / len(method_names)

    plt.figure(figsize=(12, 6))
    all_vals = []

    for i, method in enumerate(method_names):
        data, errors = grouped_data[method]
        offset = (i - len(method_names) / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, data, bar_width,
                yerr=errors if show_error_bars else None,
                capsize=4, label=method)
        all_vals.extend([
            y + e for y, e in zip(data, errors) if not np.isnan(y)
        ])
        all_vals.extend([
            y - e for y, e in zip(data, errors) if not np.isnan(y)
        ])

    if all_vals:
        y_min = min(all_vals)
        y_max = max(all_vals)
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        if plot_total_time:
            plt.ylim(0, y_max + padding)
        else:
            plt.ylim(y_min - padding, y_max + padding)

    plt.xticks(x, ks)
    plt.xlabel("k (Number of Clusters)")
    ylabel = "Normalized Avg Total Time" if plot_total_time else "Normalized Avg Cost"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. k (averaged over all datasets, m = {m_value}, normalized to '{baseline_method}')")
    plt.legend()
    plt.grid(True, axis='y')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
