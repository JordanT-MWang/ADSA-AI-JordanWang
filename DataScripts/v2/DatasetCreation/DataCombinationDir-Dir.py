import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse
import re

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
def combine_runs(parent_dir, combined_name="combinedDir"):
    parent_name = os.path.basename(os.path.normpath(parent_dir))
    combined_dir = os.path.join(parent_dir, parent_name)
    combined_edges = os.path.join(combined_dir, "Edges")

    os.makedirs(combined_edges, exist_ok=True)

    combined_input_rows = []
    combined_output_rows = []
    corrupted_images = []

    for run in sorted(os.listdir(parent_dir), key=natural_key):
        run_path = os.path.join(parent_dir, run)
        if not os.path.isdir(run_path) or run == combined_name:
            continue

        edges_dir = os.path.join(run_path, "Edges")
        input_csv = os.path.join(run_path, "input_params.csv")
        output_csv = os.path.join(run_path, "output_params.csv")

        if not (os.path.exists(edges_dir) and os.path.exists(input_csv) and os.path.exists(output_csv)):
            continue

        print(f"\nProcessing run: {run}")

        df_in = pd.read_csv(input_csv)
        df_out = pd.read_csv(output_csv)

        for img_name in tqdm(sorted(os.listdir(edges_dir), key=natural_key)):

            if not img_name.endswith(".png"):
                continue

            img_path = os.path.join(edges_dir, img_name)

            # Validate PNG
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im2:
                    width, height = im2.size
            except Exception:
                print(f"❌ CORRUPTED PNG DETECTED: {img_name} in {run}")
                corrupted_images.append((run, img_name))
                continue

            # ---- SAFE ROW LOOKUP (prevents overwrites) ----
            in_rows = df_in[df_in["Image Name"] == img_name]
            out_rows = df_out[df_out["Image Name"] == img_name]

            if in_rows.empty or out_rows.empty:
                print(f"⚠️ Missing CSV entries for {img_name}, skipping")
                continue

            # take the first match (every run should have exactly 1 per file)
            in_row = in_rows.iloc[0].copy()
            out_row = out_rows.iloc[0].copy()

            # ---- Create unique combined name ----
            prefix = run.replace(" ", "_").replace("/", "_").replace("\\", "_")
            new_name = f"{prefix}_{img_name}"
            dest_path = os.path.join(combined_edges, new_name)

            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)

            # update row entries
            in_row["Resolution"] = f"{width} x {height}"
            in_row["Source Folder"] = run
            in_row["Image Name"] = new_name

            out_row["Image Name"] = new_name

            combined_input_rows.append(in_row)
            combined_output_rows.append(out_row)


    # Write combined CSVs
    df_input_combined = (
        pd.DataFrame(combined_input_rows)
        .sort_values("Image Name", key=lambda x: x.map(natural_key))
    )

    df_output_combined = (
        pd.DataFrame(combined_output_rows)
        .sort_values("Image Name", key=lambda x: x.map(natural_key))
    )

    df_input_combined.to_csv(os.path.join(combined_dir, "input_params.csv"), index=False)
    df_output_combined.to_csv(os.path.join(combined_dir, "output_params.csv"), index=False)

    print("\n✨ Dataset combination complete!")
    print(f"Combined directory: {combined_dir}")
    print(f"❗ Corrupted PNGs detected: {len(corrupted_images)}")

    if corrupted_images:
        print("List of corrupted images:")
        for run, img in corrupted_images:
            print(f" - {run}/{img}")
def main():
    parser = argparse.ArgumentParser(description="Generates one large folder containing all images, inputs, and output parameters.")
    parser.add_argument('folder_path', type=str, help="The path of the folder containing experiment subfolders.")
    
    args = parser.parse_args()
    combine_runs(args.folder_path)


if __name__ == "__main__":
    main()