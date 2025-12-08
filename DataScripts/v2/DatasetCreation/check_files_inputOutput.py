import os
import pandas as pd
import argparse
import re

def natural_key(s):
    """Sort strings in natural numerical order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def check_csv_order(folder_path):
    edges_folder = os.path.join(folder_path, "Edges")
    input_csv = os.path.join(folder_path, "input_params.csv")
    output_csv = os.path.join(folder_path, "output_params.csv")

    # ---- Check required files ----
    if not os.path.exists(edges_folder):
        print(f"⚠️ No Edges folder in {folder_path}")
        return

    if not os.path.exists(input_csv) or not os.path.exists(output_csv):
        print(f"⚠️ Missing CSVs in {folder_path}, skipping")
        return

    # ---- Load and sort image files ----
    image_files = sorted(
        [f for f in os.listdir(edges_folder)
         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))],
        key=natural_key
    )

    #print(f"\n📂 Checking: {folder_path}")
    #print(f"   Found {len(image_files)} image(s) in Edges/")

    # ------------------------------------------------------
    # Check input_params.csv
    # ------------------------------------------------------
    df_in = pd.read_csv(input_csv)
    csv_images_in = df_in["Image Name"].tolist()

    if len(csv_images_in) != len(image_files):
        print(f"\n📂 Checking: {folder_path}")
        print(f"   ❌ INPUT count mismatch: {len(csv_images_in)} rows vs {len(image_files)} images")

    if csv_images_in == image_files[:len(csv_images_in)]:
        #print("   ✅ input_params.csv order matches")
        pass
    else:
        print(f"\n📂 Checking: {folder_path}")
        print("   ❌ input_params.csv order DOES NOT match")
        # Show first mismatch
        for i, (expected, found) in enumerate(zip(image_files, csv_images_in)):
            if expected != found:
                print(f"\n📂 Checking: {folder_path}")
                print(f"      Mismatch at row {i}: expected '{expected}' but found '{found}'")
                break

    # ------------------------------------------------------
    # Check output_params.csv
    # ------------------------------------------------------
    df_out = pd.read_csv(output_csv)
    csv_images_out = df_out["Image Name"].tolist()

    if len(csv_images_out) != len(image_files):
        print(f"\n📂 Checking: {folder_path}")
        print(f"   ❌ OUTPUT count mismatch: {len(csv_images_out)} rows vs {len(image_files)} images")

    if csv_images_out == image_files[:len(csv_images_out)]:
        #print("   ✅ output_params.csv order matches" )
        pass
    else:
        print(f"\n📂 Checking: {folder_path}")
        print("   ❌ output_params.csv order DOES NOT match")
        for i, (expected, found) in enumerate(zip(image_files, csv_images_out)):
            if expected != found:
                print(f"      Mismatch at row {i}: expected '{expected}' but found '{found}'")
                break


def main():
    parser = argparse.ArgumentParser(description="Check if input/output CSV image order matches Edges folder.")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Parent directory containing experiment folders.")
    args = parser.parse_args()

    parent_dir = args.directory

    for root, dirs, _ in os.walk(parent_dir):
        for d in dirs:
            folder_path = os.path.join(root, d)
            edges = os.path.join(folder_path, "Edges")
            input_csv = os.path.join(folder_path, "input_params.csv")
            output_csv = os.path.join(folder_path, "output_params.csv")

            if os.path.exists(edges) and os.path.exists(input_csv) and os.path.exists(output_csv):
                check_csv_order(folder_path)


if __name__ == "__main__":
    main()
