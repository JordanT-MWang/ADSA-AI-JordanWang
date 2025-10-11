import os
import pandas as pd
import argparse
"""
To run 
    python output.py /path/to/your/folder
    for my setup give the folder that contains the Edges folder
"""
def generate_output_params(folder_path):
    """
    Reads a data.dat file in the given folder and creates an output_params.csv file
    with the columns: Image Name, Surface Tension, Curvature, Area, Volume, Contact Angle.

    Assumes:
    - data.dat is tab or space delimited
    - first two lines are headers (column names and units)
    - images are in a subfolder named 'Images' with same ordering as data rows
    """
    data_path = os.path.join(folder_path, "data.dat")
    output_csv = os.path.join(folder_path, "output_params.csv")
    image_folder = os.path.join(folder_path, "Edges")

    # --- Step 1: read the data file ---
    with open(data_path, "r") as f:
        lines = f.readlines()

    # Skip first two header lines
    data_lines = lines[2:]
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

    # --- Step 2: match to image names ---
    if os.path.exists(image_folder):
        image_files = sorted(os.listdir(image_folder))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if len(image_files) != len(df):
            print(f" Warning: number of images ({len(image_files)}) != number of data rows ({len(df)})")

        df['Image Name'] = image_files[:len(df)]
    else:
        df['Image Name'] = [f"image_{i+1}.png" for i in range(len(df))]

    # --- Step 3: reorder columns and save ---
    df = df[[
        "Image Name",
        "Surface Tension (mN/m)",
        "Curvature (1/cm)",
        "Area (cm^2)",
        "Volume (ul)",
        "Contact Angle (deg)"
    ]]

    df.to_csv(output_csv, index=False)
    print(f"✅ output_params.csv saved at: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Generate a .csv file containing the output parameters matched to the images file name for all images in a folder.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing the input and data files and Edges folder.")
    
    args = parser.parse_args()

    generate_output_params(args.folder_path)

if __name__ == "__main__":
    main()