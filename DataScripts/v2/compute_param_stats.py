import os
import json
import argparse
import numpy as np
import pandas as pd

def compute_stats(dataset_path, output_json="param_stats_check.json", threshold=1e-6):
    input_csv = os.path.join(dataset_path, "input_params.csv")
    output_json = os.path.join(dataset_path, output_json)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing file: {input_csv}")

    df = pd.read_csv(input_csv)

    # Ensure the columns exist
    if "Delta Rho (g/ml)" not in df.columns:
        raise KeyError("Column 'Delta Rho (g/ml)' missing from input_params.csv")
    if "Scale Factor (cm/pixel)" not in df.columns:
        raise KeyError("Column 'Scale Factor (cm/pixel)' missing from input_params.csv")

    # Extract param matrix
    params = df[["Delta Rho (g/ml)", "Scale Factor (cm/pixel)"]].astype(np.float64).values

    # Raw stats
    mean_raw = params.mean(axis=0)
    std_raw = params.std(axis=0)

    # Detect constant parameters
    constant_mask = std_raw < threshold

    # Create safe std (avoid division by tiny values)
    std_safe = std_raw.copy()
    std_safe[constant_mask] = 1.0

    result = {
        "raw_mean": mean_raw.tolist(),
        "raw_std": std_raw.tolist(),
        "constant_parameter_mask": constant_mask.tolist(),
        "safe_mean": mean_raw.tolist(),
        "safe_std": std_safe.tolist(),
        "notes": {
            "constant_parameters_have_std_replaced_with_1.0": True,
            "threshold_used": threshold
        }
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print("\n=== Parameter Statistics ===")
    print(f"Raw Mean: {mean_raw}")
    print(f"Raw Std:  {std_raw}\n")

    print("Constant parameters detected:", constant_mask)
    print("Safe Std (after correction):", std_safe)
    print(f"\nSaved JSON to: {output_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute param stats for ADSA dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset containing input_params.csv"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="param_stats_check.json",
        help="Filename to save output JSON"
    )
    args = parser.parse_args()

    compute_stats(args.dataset_path, args.output_json)
