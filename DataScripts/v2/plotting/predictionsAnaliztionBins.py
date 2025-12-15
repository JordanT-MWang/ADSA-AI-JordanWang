import argparse
import pandas as pd
import numpy as np
import os

# ============================================================
# Core analysis function
# ============================================================
def analyze_binned_errors(
    predictions_csv_path: str,
    num_bins: int = 10,
    output_csv_path: str | None = None
):
    """
    Bin predictions by True_Value and compute:
      - Mean Absolute Error (MAE)
      - Std of Absolute Error
      - Average Prediction Time
      - Count per bin
    """

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(predictions_csv_path)

    required_cols = {
        "True_Value",
        "Predicted_Value",
        "Prediction_Time_s"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -------------------------
    # Compute absolute error
    # -------------------------
    df["abs_error"] = np.abs(df["Predicted_Value"] - df["True_Value"])

    # -------------------------
    # Create bins on True_Value
    # -------------------------
    true_min = df["True_Value"].min()
    true_max = df["True_Value"].max()

    bins = np.linspace(true_min, true_max, num_bins + 1)

    df["true_bin"] = pd.cut(
        df["True_Value"],
        bins=bins,
        include_lowest=True
    )

    # Bin centers (for Origin plotting)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_center_map = dict(zip(df["true_bin"].cat.categories, bin_centers))
    df["bin_center"] = df["true_bin"].map(bin_center_map)

    # -------------------------
    # Aggregate statistics
    # -------------------------
    binned_stats = (
        df.groupby("true_bin")
        .agg(
            MAE=("abs_error", "mean"),
            MAE_STD=("abs_error", "std"),
            Avg_Pred_Time_s=("Prediction_Time_s", "mean"),
            Pred_Time_STD_s=("Prediction_Time_s", "std"),
            Count=("abs_error", "count")
        )
        .reset_index()
    )

    # -------------------------
    # Add bin centers safely
    # -------------------------
    binned_stats["Bin_Center"] = binned_stats["true_bin"].map(bin_center_map)

    # Optional: drop interval column (Origin doesn't need it)
    binned_stats = binned_stats.drop(columns="true_bin")

    # Column order (clean for Origin)
    binned_stats = binned_stats[
        [
            "Bin_Center",
            "MAE",
            "MAE_STD",
            "Avg_Pred_Time_s",
            "Pred_Time_STD_s",
            "Count"
        ]
    ]
    # -------------------------
    # Output
    # -------------------------
    if output_csv_path is None:
        base, _ = os.path.splitext(predictions_csv_path)
        output_csv_path = f"{base}_binned_{num_bins}_bins.csv"

    binned_stats.to_csv(output_csv_path, index=False)

    print(f"\nSaved binned statistics to:\n  {output_csv_path}\n")
    print(binned_stats)


# ============================================================
# === CLI Entry Point ===
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bin CNN regression predictions by True_Value and compute MAE statistics."
    )

    parser.add_argument(
        "predictions_csv_path",
        type=str,
        help="Path to model prediction CSV file."
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=70,
        help="Number of bins for True_Value (default: 10)."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. If not provided, auto-generated."
    )

    args = parser.parse_args()

    analyze_binned_errors(
        predictions_csv_path=args.predictions_csv_path,
        num_bins=args.bins,
        output_csv_path=args.output
    )


if __name__ == "__main__":
    main()
