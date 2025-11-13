import argparse
import os
import shutil
import pandas as pd

def move_images_and_clean_input(folder_path):
    # Paths to CSVs and Edges folder
    output_csv_path = os.path.join(folder_path, "output_params.csv")
    input_csv_path = os.path.join(folder_path, "input_params.csv")
    edges_folder = os.path.join(folder_path, "Edges")
    no_params_folder = os.path.join(folder_path, "no_params")

    # Check for required files/folders
    if not os.path.isfile(output_csv_path):
        print(f"❌ output_params.csv not found in {folder_path}")
        return
    if not os.path.isfile(input_csv_path):
        print(f"❌ input_params.csv not found in {folder_path}")
        return
    if not os.path.isdir(edges_folder):
        print(f"❌ Edges folder not found in {folder_path}")
        return

    # Create no_params folder if it doesn't exist
    os.makedirs(no_params_folder, exist_ok=True)

    # Read CSVs
    output_df = pd.read_csv(output_csv_path)
    input_df = pd.read_csv(input_csv_path)

    # Get valid images from output CSV
    valid_images = set(output_df["Image Name"])

    # Move images in Edges folder that are not in output CSV
    moved_count = 0
    for img in os.listdir(edges_folder):
        img_path = os.path.join(edges_folder, img)
        if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue  # skip non-image files
        if img not in valid_images:
            shutil.move(img_path, os.path.join(no_params_folder, img))
            print(f"Moved {img} → no_params")
            moved_count += 1

    print(f"✅ Done. {moved_count} images moved to 'no_params' folder.")

    # Remove rows in input CSV without matching output CSV
    cleaned_input_df = input_df[input_df["Image Name"].isin(valid_images)]
    removed_rows = len(input_df) - len(cleaned_input_df)
    cleaned_input_df.to_csv(input_csv_path, index=False)
    print(f"✅ Removed {removed_rows} rows from input_params.csv that had no matching output entry.")


def main():
    parser = argparse.ArgumentParser(
        description="Move unmatched images to 'no_params' and clean input_params.csv."
    )
    parser.add_argument('-d', '--directory', required=True, type=str,
                        help="Path to the folder containing input_params.csv, output_params.csv, and Edges folder.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ Invalid directory path: {args.directory}")
        return

    move_images_and_clean_input(args.directory)


if __name__ == "__main__":
    main()
