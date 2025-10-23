import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_predictions(csv_path):
    # === Load Data ===
    data = pd.read_csv(csv_path)
    true_values = data["True_Value"].values
    predicted_values = data["Predicted_Value"].values
    times = data["Prediction_Time_s"].values

    # === Compute Metrics ===
    mae = mean_absolute_error(true_values, predicted_values)
    relative_errors = np.abs((true_values - predicted_values) / true_values)
    accuracy = (1 - np.mean(relative_errors)) * 100
    r2 = r2_score(true_values, predicted_values)
    avg_time = np.mean(times)

    # === Fit Line of Best Fit ===
    m, b = np.polyfit(true_values, predicted_values, 1)
    line_best_fit = m * np.array(true_values) + b

    # === Plot ===
    plt.figure(figsize=(7, 7))
    plt.scatter(true_values, predicted_values, label="Predictions", alpha=0.7)
    plt.plot(true_values, true_values, "k--", label="Perfect 45° Line (y = x)")
    plt.plot(true_values, line_best_fit, "r-", label=f"Best Fit Line (y = {m:.2f}x + {b:.2f})")

    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Predicted vs True Values")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # === Add Text with Metrics (below the x-axis) ===
    textstr = (
        f"MAE: {mae:.4f}   "
        f"Accuracy: {accuracy:.2f}%   "
        f"R²: {r2:.4f}   "
        f"Avg Time: {avg_time:.4f}s"
    )
    # Add centered text below the x-axis
    plt.gcf().text(
        0.5, -0.05, textstr,
        ha="center", va="center",
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
    )

    # === Save Plot ===
    graphs_dir = os.path.join(os.path.dirname(csv_path), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(csv_path))[0] + "_analysis.png"
    output_path = os.path.join(graphs_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graph saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze predicted vs true values from a CSV file."
    )
    parser.add_argument("output_params_path", type=str, help="Path to the output_params.csv.")
    args = parser.parse_args()

    analyze_predictions(args.output_params_path)

if __name__ == "__main__":
    main()
