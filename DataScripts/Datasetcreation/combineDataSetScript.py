import os
import shutil
import pandas as pd
import argparse

def combine_datasets(parent_folder):
    """
    Combines all child experiment folders (each containing input_params.csv, output_params.csv, and Edges/)
    into a single dataset under 'DataSetCombined' inside the parent folder.

    - Skips any rows where the corresponding image file is missing.
    - Works even if input/output CSVs are misaligned.
    """
    combined_dir = os.path.join(os.path.dirname(parent_folder), "DataSetCombined")
    os.makedirs(combined_dir, exist_ok=True)

    combined_edges_dir = os.path.join(combined_dir, "Edges")
    os.makedirs(combined_edges_dir, exist_ok=True)

    combined_input_csv = os.path.join(combined_dir, "input_params.csv")
    combined_output_csv = os.path.join(combined_dir, "output_params.csv")

    input_dfs = []
    output_dfs = []
    image_counter = 1  # used for renaming

    summary = []

    # --- Loop through all child folders ---
    for child_name in sorted(os.listdir(parent_folder)):
        child_path = os.path.join(parent_folder, child_name)
        if not os.path.isdir(child_path):
            continue

        input_csv = os.path.join(child_path, "input_params.csv")
        output_csv = os.path.join(child_path, "output_params.csv")
        edges_folder = os.path.join(child_path, "Edges")

        if not (os.path.exists(input_csv) and os.path.exists(output_csv) and os.path.exists(edges_folder)):
            print(f"Skipping {child_name}: missing required files/folders")
            continue

        print(f"Processing {child_name} ...")

        # Read CSVs
        input_df = pd.read_csv(input_csv)
        output_df = pd.read_csv(output_csv)

        rename_map = {}
        copied_count = 0
        skipped_count = 0

        # --- Check each image and copy it if it exists ---
        for old_name in input_df["Image Name"]:
            old_path = os.path.join(edges_folder, old_name)
            if os.path.exists(old_path):
                base, ext = os.path.splitext(old_name)
                new_name = f"{child_name}_{image_counter:05d}{ext}"
                rename_map[old_name] = new_name

                new_path = os.path.join(combined_edges_dir, new_name)
                shutil.copy2(old_path, new_path)
                image_counter += 1
                copied_count += 1
            else:
                print(f"⚠️ Missing image, skipping row: {old_path}")
                skipped_count += 1

        # Skip folder if no valid images
        if copied_count == 0:
            print(f"⚠️ No valid images found in {child_name}, skipping folder.")
            continue

        # --- Keep only rows corresponding to existing images ---
        input_df_valid = input_df[input_df["Image Name"].isin(rename_map.keys())].copy()
        output_df_valid = output_df[output_df["Image Name"].isin(rename_map.keys())].copy()

        # Rename Image Names
        input_df_valid["Image Name"] = input_df_valid["Image Name"].map(rename_map)
        output_df_valid["Image Name"] = output_df_valid["Image Name"].map(rename_map)

        # Add source folder info
        input_df_valid["Source Folder"] = child_name
        output_df_valid["Source Folder"] = child_name

        input_dfs.append(input_df_valid)
        output_dfs.append(output_df_valid)

        summary.append({
            "folder": child_name,
            "copied": copied_count,
            "skipped": skipped_count
        })

    # --- Combine all and save ---
    if input_dfs:
        combined_input = pd.concat(input_dfs, ignore_index=True)
        combined_input.to_csv(combined_input_csv, index=False)
        print(f"✅ Combined input saved: {combined_input_csv}")

    if output_dfs:
        combined_output = pd.concat(output_dfs, ignore_index=True)
        combined_output.to_csv(combined_output_csv, index=False)
        print(f"✅ Combined output saved: {combined_output_csv}")

    print(f"All images copied to: {combined_edges_dir}")
    print("\nSummary per folder:")
    for s in summary:
        print(f" {s['folder']}: {s['copied']} copied, {s['skipped']} skipped")

    print("Dataset combination complete.")


def main():
    parser = argparse.ArgumentParser(description="Generates one large folder containing all images, inputs, and output parameters.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing experiment subfolders.")
    
    args = parser.parse_args()
    combine_datasets(args.folder_path)


if __name__ == "__main__":
    main()
