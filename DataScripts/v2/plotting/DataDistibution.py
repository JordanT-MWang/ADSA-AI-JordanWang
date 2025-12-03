import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import re

def analyze_distribution(csv_path, input_csv_path=None, bin_sizes=None):
    """
    Analyzes the distribution of output and input parameters, including scale factor,
    image size, and image aspect ratio.
    python .\DataScripts\plotting\DataDistibution.py "E:\Data\Jordan\Jordan Wang AI-ADSA\Dataset\DataSetCombined\SurfaceTension\black_edges\output_params.csv" --input_params_path "E:\Data\Jordan\Jordan Wang AI-ADSA\Dataset\DataSetCombined\SurfaceTension\black_edges\input_params.csv"
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Output CSV not found: {csv_path}")

    df_out = pd.read_csv(csv_path)
    
    # === Output Parameter Columns ===
    expected_output_cols = [
        "Surface Tension (mN/m)",
        "Curvature (1/cm)",
        "Area (cm^2)",
        "Volume (ul)",
        "Contact Angle (deg)",
        "Source Folder"
    ]

    # Default bin sizes
    if bin_sizes is None:
        bin_sizes = {
            "Density"
            "Surface Tension (mN/m)": 5,
            "Curvature (1/cm)": 0.5,
            "Area (cm^2)": 0.05,
            "Volume (ul)": 1.0,
            "Contact Angle (deg)": 5,
            "Scale Factor (cm/pixel)": 0.001,
            "Image Size (pixels)": 200000,
            "Aspect Ratio": 0.05
        }

    out_dir = os.path.dirname(csv_path)

    # === Analyze output_params.csv ===
    for col in expected_output_cols:
        if col in df_out.columns:
            _plot_distribution(df_out[col], col, bin_sizes.get(col, 1), out_dir)
        else:
            print(f"⚠️  Skipping missing output column: {col}")

    # === Analyze input_params.csv (optional) ===
    if input_csv_path and os.path.exists(input_csv_path):
        print("\n📂 Including input parameters...")
        df_in = pd.read_csv(input_csv_path)

        # Scale Factor
        if "Scale Factor (cm/pixel)" in df_in.columns:
            _plot_distribution(
                df_in["Scale Factor (cm/pixel)"],
                "Scale Factor (cm/pixel)",
                bin_sizes.get("Scale Factor (cm/pixel)", 0.001),
                out_dir
            )
        else:
            print("⚠️  Missing 'Scale Factor (cm/pixel)' column in input CSV.")
        # Delta Rho histogram (reuses existing code)
        if "Delta Rho (g/ml)" in df_in.columns:
            _plot_distribution(
                df_in["Delta Rho (g/ml)"],
                "Delta Rho (g/ml)",
                bin_sizes.get("Delta Rho (g/ml)", 0.01),
                out_dir
            )
        else:
            print("⚠️ Missing 'Delta Rho (g/ml)' column in input CSV.")
        # Resolution → parse width/height
        if "Resolution" in df_in.columns:
            df_in["Width"], df_in["Height"] = zip(*df_in["Resolution"].apply(_parse_resolution))
            df_in["Image Size (pixels)"] = df_in["Width"] * df_in["Height"]
            df_in["Aspect Ratio"] = df_in["Width"] / df_in["Height"]

            # Image Size histogram
            _plot_distribution(
                df_in["Image Size (pixels)"].dropna(),
                "Image Size (pixels)",
                bin_sizes.get("Image Size (pixels)", 200000),
                out_dir
            )

            # Aspect Ratio histogram
            _plot_distribution(
                df_in["Aspect Ratio"].dropna(),
                "Aspect Ratio (W/H)",
                bin_sizes.get("Aspect Ratio", 0.05),
                out_dir
            )

        else:
            print("⚠️  Missing 'Resolution' column in input CSV.")

    else:
        print("\n⚠️  No input_params.csv provided or found.")

    print("\n✅ Analysis complete for all parameters!")


# === Helper: parse resolution into width and height ===
def _parse_resolution(res_str):
    if isinstance(res_str, str):
        match = re.match(r"(\d+)\s*x\s*(\d+)", res_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan


def _plot_distribution(values, label, bin_size=None, out_dir=None, min_bins=20):
    """
    Plot a histogram or bar chart for numeric or categorical data.
    Automatically detects type:
      - Numeric with many unique values → histogram
      - Categorical or few unique values → bar chart
    """
    values = pd.Series(values).dropna()

    # Detect categorical: non-numeric or few unique values
    num_unique = values.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(values)
    if not is_numeric or num_unique <= 20:
        # Categorical / few unique values → horizontal bar chart
        counts = values.value_counts().sort_index()
        num_categories = len(counts)
        plt.figure(figsize=(max(10, num_categories*0.4), max(6, num_categories*0.25)))  # scale figure size
        plt.barh(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.7)
        plt.ylabel(label)
        plt.xlabel("Number of Samples")
        plt.title(f"Distribution of {label} (categorical)")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

    else:
        # Numeric → histogram
        values = pd.to_numeric(values, errors="coerce").dropna()
        values = values[values > 0]
        if len(values) == 0:
            print(f"⚠️  No valid values for {label}, skipping.")
            return

        min_val, max_val = values.min(), values.max()
        range_val = max_val - min_val

        if range_val == 0:
            min_val *= 0.999
            max_val *= 1.001
            range_val = max_val - min_val

        if bin_size is None:
            bin_size = max(range_val / 20, range_val / min_bins)

        num_bins = max(int(np.ceil(range_val / bin_size)), min_bins)
        bins = np.linspace(min_val, max_val, num_bins + 1)
        counts, edges = np.histogram(values, bins=bins)

        plt.figure(figsize=(9, 5))
        plt.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            width=(edges[1] - edges[0]) * 0.9,
            edgecolor="black",
            alpha=0.7
        )
        plt.xlabel(label)
        plt.ylabel("Number of Samples")
        plt.title(f"Distribution of {label}")
        plt.grid(alpha=0.3)

    # Save plot
    if out_dir is not None:
        save_path = os.path.join(out_dir, f"distribution_{label.replace('/', '_').replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  💾 Saved plot: {save_path}")
    else:
        plt.show()


# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser(description="Analyze dataset distribution for output and input parameters.")
    parser.add_argument("output_params_path", type=str, help="Path to the output_params.csv.")
    parser.add_argument("--input_params_path", type=str, help="Optional path to input_params.csv.")
    args = parser.parse_args()

    analyze_distribution(args.output_params_path, input_csv_path=args.input_params_path)


if __name__ == "__main__":
    main()
