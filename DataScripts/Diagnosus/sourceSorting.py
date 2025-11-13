import os
import shutil
import pandas as pd
import argparse

def run_edges(parent_dir, output_dir_name="run_edges"):
    """
    Organize edges by source folder into a new directory and copy filtered CSVs.
    """
    edges_dir = os.path.join(parent_dir, "Edges")
    input_csv_path = os.path.join(parent_dir, "input_params.csv")
    output_csv_path = os.path.join(parent_dir, "output_params.csv")

    # Validate
    if not os.path.exists(edges_dir):
        raise FileNotFoundError(f"Edges folder not found in {parent_dir}")
    if not os.path.exists(input_csv_path) or not os.path.exists(output_csv_path):
        raise FileNotFoundError("input_params.csv or output_params.csv not found in parent directory")

    # Load CSVs
    input_df = pd.read_csv(input_csv_path)
    output_df = pd.read_csv(output_csv_path)

    # Create the new output directory
    run_edges_dir = os.path.join(parent_dir, output_dir_name)
    os.makedirs(run_edges_dir, exist_ok=True)

    # Map image name → source folder
    source_map = dict(zip(output_df["Image Name"], output_df["Source Folder"]))

    # Get all images in Edges folder
    image_files = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    if not image_files:
        print("No images found in Edges folder")
        return

    print(f"Processing {len(image_files)} images...")

    # Organize by source folder
    for img_file in image_files:
        if img_file not in source_map:
            print(f"⚠️  {img_file} not found in output_params.csv, skipping")
            continue

        source_folder = str(source_map[img_file])
        dest_source_dir = os.path.join(run_edges_dir, source_folder)
        dest_edges_dir = os.path.join(dest_source_dir, "Edges")
        os.makedirs(dest_edges_dir, exist_ok=True)

        # Copy image
        shutil.copy2(
            os.path.join(edges_dir, img_file),
            os.path.join(dest_edges_dir, img_file)
        )

    # Copy filtered CSVs for each source folder
    for source in output_df["Source Folder"].unique():
        dest_source_dir = os.path.join(run_edges_dir, str(source))
        dest_edges_dir = os.path.join(dest_source_dir, "Edges")
        if not os.path.exists(dest_edges_dir):
            continue  # skip empty source folders

        # Filter CSVs for this source
        input_filtered = input_df[input_df["Source Folder"] == source]
        output_filtered = output_df[output_df["Source Folder"] == source]

        # Keep only images that exist in the edges folder
        existing_images = os.listdir(dest_edges_dir)
        input_filtered = input_filtered[input_filtered["Image Name"].isin(existing_images)]
        output_filtered = output_filtered[output_filtered["Image Name"].isin(existing_images)]

        # Save filtered CSVs
        input_filtered.to_csv(os.path.join(dest_source_dir, "input_params.csv"), index=False)
        output_filtered.to_csv(os.path.join(dest_source_dir, "output_params.csv"), index=False)

    print(f"\n✅ Finished organizing edges into {run_edges_dir}")


def main():
    parser = argparse.ArgumentParser(description="Organize Edges folder into source folders with filtered CSVs.")
    parser.add_argument("parent_dir", type=str, help="Parent directory containing Edges and CSVs.")
    parser.add_argument("--output_dir_name", type=str, default="run_edges", help="Name for the new output directory.")
    args = parser.parse_args()

    run_edges(args.parent_dir, args.output_dir_name)


if __name__ == "__main__":
    main()
