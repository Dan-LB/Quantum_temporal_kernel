import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def read_accuracy(file_path):
    with open(file_path, "r") as f:
        return float(f.readline().strip().split(": ")[1])

def load_results(results_folder="exps_results"):
    data = []
    for exp_folder in os.listdir(results_folder):
        encoding = exp_folder.split("__")[1]
        exp_type = exp_folder.split("__")[0]
        exp_path = os.path.join(results_folder, exp_folder)
        
        for seed_folder in os.listdir(exp_path):
            seed_path = os.path.join(exp_path, seed_folder)
            
            train_acc = read_accuracy(os.path.join(seed_path, "train_accuracy.txt"))
            val_acc = read_accuracy(os.path.join(seed_path, "val_accuracy.txt"))
            test_acc = read_accuracy(os.path.join(seed_path, "test_accuracy.txt"))

            data.append({
                "Experiment": exp_type,
                "Encoding": encoding,
                "Seed": seed_folder.split("_")[1],
                "Train Accuracy": train_acc,
                "Validation Accuracy": val_acc,
                "Test Accuracy": test_acc
            })
    
    return pd.DataFrame(data)

# Load results into a DataFrame
df = load_results()

# Define colors
color_Q_A = (0.1, 0.6, 0.8)
color_Q_B = (0.2, 0.8, 0.6)
color_C_A = (0.8, 0.3, 0.1)
color_C_B = (0.9, 0.6, 0.2)

# Split data into classical and non-classical experiments
classical_df = df[df["Experiment"].str.startswith("Classical")]
non_classical_df = df[~df["Experiment"].str.startswith("Classical")]

# Group by Experiment and Encoding, calculate mean, std, and sample count for each metric
def summarize_results(dataframe):
    return dataframe.groupby(["Experiment", "Encoding"]).agg(
        Train_Mean=("Train Accuracy", "mean"),
        Train_Std=("Train Accuracy", "std"),
        Validation_Mean=("Validation Accuracy", "mean"),
        Validation_Std=("Validation Accuracy", "std"),
        Test_Mean=("Test Accuracy", "mean"),
        Test_Std=("Test Accuracy", "std"),
        Sample_Count=("Train Accuracy", "size")
    ).reset_index()

classical_summary = summarize_results(classical_df)
non_classical_summary = summarize_results(non_classical_df)

# Create subplots for classical and non-classical experiments
fig, (ax_non_classical, ax_classical) = plt.subplots(2, 1, figsize=(12, 12), sharex=False)
metrics = ["Train", "Validation", "Test"]
bar_width = 0.2

# Helper function to plot data
def plot_summary(ax, summary, color_A, color_B):
    x = np.arange(len(summary))  # Label locations
    offsets = [-bar_width, 0, bar_width]

    for i, metric in enumerate(metrics):
        for idx, row in summary.iterrows():
            # Determine color for the metric
            if metric == "Train":
                color = color_A
            elif metric == "Validation":
                color = tuple((np.array(color_A) + np.array(color_B)) / 2)
            else:
                color = color_B
            
            # Determine transparency and display sample count if needed
            alpha = 0.7 if row["Sample_Count"] < 10 else 1.0
            ax.bar(
                x[idx] + offsets[i],
                row[f"{metric}_Mean"],
                width=bar_width,
                yerr=row[f"{metric}_Std"],
                capsize=5,
                color=color,
                alpha=alpha,
                label=f"{metric} Accuracy" if idx == 0 else ""
            )
            
            # Add sample count if below 10 samples
            if row["Sample_Count"] < 10:
                ax.text(
                    x[idx] + offsets[i],
                    row[f"{metric}_Mean"] + row[f"{metric}_Std"] + 0.01,
                    str(row["Sample_Count"]),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black"
                )

    ax.set_ylim(0.45, 1.1)
    ax.legend()

# Plot non-classical experiments on the upper row
plot_summary(ax_non_classical, non_classical_summary, color_Q_A, color_Q_B)
ax_non_classical.set_title("Non-Classical Experiments")
ax_non_classical.set_ylabel("Accuracy")

# Set x-tick labels for the non-classical experiments (upper row)
x_labels_non_classical = non_classical_summary["Experiment"] + " - " + non_classical_summary["Encoding"]
x_non_classical = np.arange(len(non_classical_summary))
ax_non_classical.set_xticks(x_non_classical)
ax_non_classical.set_xticklabels(x_labels_non_classical, rotation=15, ha="right")


# Plot classical experiments on the bottom row
plot_summary(ax_classical, classical_summary, color_C_A, color_C_B)
ax_classical.set_title("Classical Experiments")
ax_classical.set_xlabel("Experiment and Encoding")
ax_classical.set_ylabel("Accuracy")

# Set x-tick labels for the classical experiments (bottom row)
x_labels_classical = classical_summary["Experiment"] + " - " + classical_summary["Encoding"]
x_classical = np.arange(len(classical_summary))
ax_classical.set_xticks(x_classical)
ax_classical.set_xticklabels(x_labels_classical, rotation=15, ha="right")

#plt.tight_layout()
plt.show()
