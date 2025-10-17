import os
import pandas as pd
import argparse

"""
To run:
    python output.py -d /path/to/parent_folder
This will run the script for every child directory that contains data.dat and an Edges folder.
"""

def generate_output_params(folder_path):
    """
    Reads a data.dat file in the given folder and creates an output_params.csv file
    with the columns: Image Name, Surface Tension, Curvature, Area, Volume, Contact Angle.
    """
    data_path = os.path.join(folder_path, "data.dat")
    image_folder = os.path.join(folder_path, "Edges")
    output_csv = os.path.join(folder_path, "output_params.csv")

    # --- Step 1: Check required files ---
    if not os.path.exists(data_path):
        print(f"⚠️ Skipping {folder_path} — data.dat not found.")
        return
    if not os.path.exists(image_folder):
        print(f"⚠️ Skipping {folder_path} — Edges folder not found.")
        return

    # --- Step 2: Read the data file ---
    with open(data_path, "r") as f:
        lines = f.readlines()

    data_lines = lines[2:]  # Skip header lines
    data = []

    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split()
        # Expected format: Number, Time, SurT, Curv, Area, Volu, CAgl
        if len(parts) >= 7:
            data.append({
                "Surface Tension (mN/m)": float(parts[2]),
                "Curvature (1/cm)": float(parts[3]),
                "Area (cm^2)": float(parts[4]),
                "Volume (ul)": float(parts[5]),
                "Contact Angle (deg)": float(parts[6]),
            })

    df = pd.DataFrame(data)

    # --- Step 3: Match to image names ---
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if len(image_files) != len(df):
        print(f"⚠️ {folder_path}: {len(image_files)} images vs {len(df)} data rows (mismatch)")

    df['Image Name'] = image_files[:len(df)]

    # --- Step 4: Reorder columns and save ---
    df = df[[
        "Image Name",
        "Surface Tension (mN/m)",
        "Curvature (1/cm)",
        "Area (cm^2)",
        "Volume (ul)",
        "Contact Angle (deg)"
    ]]

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate output_params.csv for all folders containing data.dat and Edges.")
    parser.add_argument('-d', '--directory', type=str, required=True, help="Path to the parent directory containing experiment folders.")
    args = parser.parse_args()

    parent_dir = args.directory

    for root, dirs, _ in os.walk(parent_dir):
        for d in dirs:
            folder_path = os.path.join(root, d)
            if os.path.exists(os.path.join(folder_path, "data.dat")) and os.path.exists(os.path.join(folder_path, "Edges")):
                generate_output_params(folder_path)


if __name__ == "__main__":
    main()
