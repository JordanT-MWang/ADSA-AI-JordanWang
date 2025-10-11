import os
import shutil
import pandas as pd
import argparse

def combine_datasets(parent_folder):
    """
    Combines all child experiment folders (each containing input_params.csv, output_params.csv, and Edges/)
    into a single dataset under 'DataSetCombined' inside the parent folder.
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

        # --- Create a mapping from old image names to new image names ---
        rename_map = {}
        for old_name in input_df["Image Name"]:
            base, ext = os.path.splitext(old_name)
            new_name = f"{child_name}_{image_counter:05d}{ext}"
            rename_map[old_name] = new_name

            old_path = os.path.join(edges_folder, old_name)
            new_path = os.path.join(combined_edges_dir, new_name)
            if os.path.exists(old_path):
                shutil.copy2(old_path, new_path)
            else:
                print(f" Missing image: {old_path}")
            image_counter += 1

        # Rename images in CSVs using the mapping
        input_df["Image Name"] = input_df["Image Name"].map(rename_map)
        output_df["Image Name"] = output_df["Image Name"].map(rename_map)

        # Add source info (optional but useful)
        input_df["Source Folder"] = child_name
        output_df["Source Folder"] = child_name

        input_dfs.append(input_df)
        output_dfs.append(output_df)

    # --- Combine all and save ---
    if input_dfs:
        combined_input = pd.concat(input_dfs, ignore_index=True)
        combined_input.to_csv(combined_input_csv, index=False)
        print(f"Combined input saved: {combined_input_csv}")

    if output_dfs:
        combined_output = pd.concat(output_dfs, ignore_index=True)
        combined_output.to_csv(combined_output_csv, index=False)
        print(f"Combined output saved: {combined_output_csv}")

    print(f"All images copied to: {combined_edges_dir}")
    print("Dataset combination complete.")


def main():
    parser = argparse.ArgumentParser(description="Generates one large folder containing all images, inputs, and output parameters.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing the input and data files and Edges folder.")
    
    args = parser.parse_args()
    combine_datasets(args.folder_path)


if __name__ == "__main__":
    main()
