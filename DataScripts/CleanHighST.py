import argparse
import os
import pandas as pd

def remove_high_surface_tension(folder_path, threshold=75):
    # Paths to CSVs and Edges folder
    output_csv_path = os.path.join(folder_path, "output_params.csv")
    input_csv_path = os.path.join(folder_path, "input_params.csv")
    edges_folder = os.path.join(folder_path, "Edges")

    # Check if files exist
    if not os.path.isfile(output_csv_path):
        print(f"❌ output_params.csv not found in {folder_path}")
        return
    if not os.path.isfile(input_csv_path):
        print(f"❌ input_params.csv not found in {folder_path}")
        return
    if not os.path.isdir(edges_folder):
        print(f"❌ Edges folder not found in {folder_path}")
        return

    # Read CSVs
    output_df = pd.read_csv(output_csv_path)
    input_df = pd.read_csv(input_csv_path)

    # Find images with surface tension > threshold
    high_st_df = output_df[output_df["Surface Tension (mN/m)"] > threshold]

    if high_st_df.empty:
        print("No images with surface tension above threshold found.")
        return

    images_to_remove = high_st_df["Image Name"].tolist()
    print(f"Removing {len(images_to_remove)} images with Surface Tension > {threshold} mN/m")

    # Remove from output CSV
    output_df = output_df[output_df["Surface Tension (mN/m)"] <= threshold]

    # Remove from input CSV
    input_df = input_df[~input_df["Image Name"].isin(images_to_remove)]

    # Delete images from Edges folder
    for img in images_to_remove:
        img_path = os.path.join(edges_folder, img)
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted {img_path}")
        else:
            print(f"Image not found: {img_path}")

    # Save updated CSVs
    output_df.to_csv(output_csv_path, index=False)
    input_df.to_csv(input_csv_path, index=False)
    print("✅ CSV files updated.")


def main():
    parser = argparse.ArgumentParser(
        description="Remove images with high surface tension (>75 mN/m) and update CSVs."
    )
    parser.add_argument('-d', '--directory', required=True, type=str,
                        help="Path to the folder containing input_params.csv, output_params.csv, and Edges folder.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ Invalid directory path: {args.directory}")
        return

    remove_high_surface_tension(args.directory)


if __name__ == "__main__":
    main()
