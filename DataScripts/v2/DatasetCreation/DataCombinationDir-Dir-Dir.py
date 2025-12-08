import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse


def find_runs(root_path):
    """
    Recursively yields any directory containing:
      - Edges/
      - input_params.csv
      - output_params.csv
    """
    for current_root, dirs, files in os.walk(root_path):
        edges = os.path.join(current_root, "Edges")
        input_csv = os.path.join(current_root, "input_params.csv")
        output_csv = os.path.join(current_root, "output_params.csv")

        if os.path.isdir(edges) and os.path.exists(input_csv) and os.path.exists(output_csv):
            yield current_root


def combine_runs(parent_dir):
    parent_name = os.path.basename(os.path.normpath(parent_dir))
    combined_dir = os.path.join(parent_dir, parent_name)
    combined_edges = os.path.join(combined_dir, "Edges")

    os.makedirs(combined_edges, exist_ok=True)

    combined_input_rows = []
    combined_output_rows = []
    corrupted_images = []

    print(f"\n🔍 Searching for runs under: {parent_dir}\n")

    # ---------------------------------------------------------
    # Find all runs at any depth
    # ---------------------------------------------------------
    all_runs = list(find_runs(parent_dir))

    print(f"📁 Found {len(all_runs)} valid run(s).")
    for r in all_runs:
        print(" -", r)

    # ---------------------------------------------------------
    # Iterate through each discovered run
    # ---------------------------------------------------------
    for run_path in all_runs:
        run_name = os.path.relpath(run_path, parent_dir).replace("\\", "/")
        print(f"\nProcessing run: {run_name}")

        edges_dir = os.path.join(run_path, "Edges")
        input_csv = os.path.join(run_path, "input_params.csv")
        output_csv = os.path.join(run_path, "output_params.csv")

        df_in = pd.read_csv(input_csv).set_index("Image Name")
        df_out = pd.read_csv(output_csv).set_index("Image Name")

        for img_name in tqdm(os.listdir(edges_dir)):
            if not img_name.endswith(".png"):
                continue

            img_path = os.path.join(edges_dir, img_name)

            # --- Validate PNG ---
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im2:
                    width, height = im2.size
            except Exception:
                print(f"❌ CORRUPTED PNG DETECTED: {img_name} in {run_name}")
                corrupted_images.append((run_name, img_name))
                continue

            # Missing CSV rows?
            if img_name not in df_in.index or img_name not in df_out.index:
                print(f"⚠️ Missing CSV entries for {img_name} in {run_name}, skipping")
                continue

            # Make global unique image name using full relative path
            prefix = run_name.replace("/", "_").replace(" ", "_")
            new_name = f"{prefix}_{img_name}"
            dest_path = os.path.join(combined_edges, new_name)

            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)

            # INPUT row
            in_row = df_in.loc[img_name].to_dict()
            in_row["Resolution"] = f"{width} x {height}"
            in_row["Source Folder"] = run_name
            in_row["Image Name"] = new_name
            combined_input_rows.append(in_row)

            # OUTPUT row
            out_row = df_out.loc[img_name].to_dict()
            out_row["Image Name"] = new_name
            combined_output_rows.append(out_row)

    # ---------------------------------------------------------
    # Write final combined CSVs
    # ---------------------------------------------------------
    df_input_combined = pd.DataFrame(combined_input_rows).sort_values("Image Name")
    df_output_combined = pd.DataFrame(combined_output_rows).sort_values("Image Name")

    df_input_combined.to_csv(os.path.join(combined_dir, "input_params.csv"), index=False)
    df_output_combined.to_csv(os.path.join(combined_dir, "output_params.csv"), index=False)

    print("\n✨ Dataset combination complete!")
    print(f"📂 Combined directory: {combined_dir}")
    print(f"❗ Corrupted PNGs detected: {len(corrupted_images)}")

    if corrupted_images:
        print("\nCorrupted image list:")
        for run, img in corrupted_images:
            print(f" - {run}/{img}")


def main():
    parser = argparse.ArgumentParser(description="Combine multiple experiment runs into a unified dataset.")
    parser.add_argument("folder_path", type=str, help="Top-level folder containing nested experiment directories.")
    args = parser.parse_args()
    combine_runs(args.folder_path)


if __name__ == "__main__":
    main()
