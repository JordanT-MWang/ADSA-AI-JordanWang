import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
def analyze_distribution(csv_path, bin_sizes=None):
    """
    Analyzes the distribution of all physical parameters in output_params.csv.

    Parameters:
        csv_path (str): Path to the combined output_params.csv file.
        bin_sizes (dict): Optional custom bin sizes for each parameter.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expected columns
    expected_columns = [
        "Surface Tension (mN/m)",
        "Curvature (1/cm)",
        "Area (cm^2)",
        "Volume (ul)",
        "Contact Angle (deg)"
    ]

    # Default bin sizes (tuned for physical ranges)
    if bin_sizes is None:
        bin_sizes = {
            "Surface Tension (mN/m)": 5,
            "Curvature (1/cm)": 0.5,
            "Area (cm^2)": 0.05,
            "Volume (ul)": 1.0,
            "Contact Angle (deg)": 5
        }

    out_dir = os.path.dirname(csv_path)

    for col in expected_columns:
        if col not in df.columns:
            print(f"⚠️  Skipping missing column: {col}")
            continue

        print("\n" + "=" * 60)
        print(f"📊 Analyzing: {col}")

        # Clean up data
        values = df[col].dropna().to_numpy()
        values = values[values > 0]  # ignore invalid or zero values
        if len(values) == 0:
            print("⚠️  No valid values found, skipping.")
            continue

        bin_size = bin_sizes[col]
        max_val = np.ceil(values.max())
        bins = np.arange(0, max_val + bin_size, bin_size)
        counts, edges = np.histogram(values, bins=bins)

        # Print summary
        print(f"  Samples: {len(values)}")
        print(f"  Range: {values.min():.3f} – {values.max():.3f}")
        print(f"  Bin size: {bin_size}")
        print(f"  Num bins: {len(bins)-1}")

        # Bin distribution
        print("  Bin distribution:")
        for i in range(len(counts)):
            print(f"    {edges[i]:8.3f}–{edges[i+1]:8.3f} : {counts[i]:6d}")

        # Balance stats
        mean_count = np.mean(counts)
        imbalance_ratio = counts.max() / (counts.min() + 1e-9)
        print(f"\n  ⚖️  Mean per-bin count: {mean_count:.1f}")
        print(f"  📈  Max/Min bin ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("  ⚠️  Dataset is quite imbalanced — consider balancing bins.")
        elif imbalance_ratio > 1.5:
            print("  ℹ️  Dataset is moderately imbalanced.")
        else:
            print("  ✅  Dataset looks balanced!")

        # Plot histogram
        plt.figure(figsize=(9, 5))
        plt.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            width=bin_size * 0.9,
            edgecolor="black",
            alpha=0.7
        )
        plt.xlabel(col)
        plt.ylabel("Number of Samples")
        plt.title(f"Distribution of {col}")
        plt.grid(alpha=0.3)

        # Save the figure
        save_path = os.path.join(out_dir, f"distribution_{col.replace('/', '_').replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"  💾 Saved plot: {save_path}")

    print("\n✅ Analysis complete for all columns!")

# Example usage:
# analyze_surface_tension_distribution("/path/to/DataSetCombined/output_params.csv", bin_size=5)
def main():
    parser = argparse.ArgumentParser(
        description="checks data dirstuption of data set."
    )
    parser.add_argument("output_params_path", type=str, help="Path to the output_params.csv.")
    args = parser.parse_args()
    
    analyze_distribution(args.output_params_path)


if __name__ == "__main__":
    main()