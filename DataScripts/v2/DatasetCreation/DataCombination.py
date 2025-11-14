import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

def combine_runs(parent_dir, combined_name="combinedDir"):
    combined_dir = os.path.join(parent_dir, combined_name)
    combined_edges = os.path.join(combined_dir, "Edges")

    os.makedirs(combined_edges, exist_ok=True)

    combined_input_rows = []
    combined_output_rows = []
    corrupted_images = []

    for run in sorted(os.listdir(parent_dir)):
        run_path = os.path.join(parent_dir, run)
        if not os.path.isdir(run_path) or run == combined_name:
            continue

        edges_dir = os.path.join(run_path, "Edges")
        input_csv = os.path.join(run_path, "input_params.csv")
        output_csv = os.path.join(run_path, "output_params.csv")

        if not (os.path.exists(edges_dir) and os.path.exists(input_csv) and os.path.exists(output_csv)):
            continue

        print(f"\nProcessing run: {run}")

        df_in = pd.read_csv(input_csv).set_index("Image Name")
        df_out = pd.read_csv(output_csv).set_index("Image Name")

        for img_name in tqdm(os.listdir(edges_dir)):
            if not img_name.endswith(".png"):
                continue

            img_path = os.path.join(edges_dir, img_name)

            # --- Validate PNG corruption ---
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im2:
                    width, height = im2.size
            except Exception:
                print(f"❌ CORRUPTED PNG DETECTED: {img_name} in {run}")
                corrupted_images.append((run, img_name))
                continue

            # Skip if CSVs missing the row
            if img_name not in df_in.index or img_name not in df_out.index:
                print(f"⚠️ Missing CSV entries for {img_name}, skipping")
                continue

            # -------------------------------
            # Create unique global image name
            # -------------------------------
            prefix = run.replace(" ", "_").replace("/", "_").replace("\\", "_")
            new_name = f"{prefix}_{img_name}"
            dest_path = os.path.join(combined_edges, new_name)

            # Copy image with NEW NAME (avoids overwrite)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)

            # Add INPUT row
            in_row = df_in.loc[img_name].copy()
            in_row["Resolution"] = f"{width} x {height}"
            in_row["Source Folder"] = run
            in_row["Image Name"] = new_name
            combined_input_rows.append(in_row)

            # Add OUTPUT row
            out_row = df_out.loc[img_name].copy()
            out_row["Image Name"] = new_name
            combined_output_rows.append(out_row)

    # Write combined CSVs
    df_input_combined = pd.DataFrame(combined_input_rows).sort_values("Image Name")
    df_output_combined = pd.DataFrame(combined_output_rows).sort_values("Image Name")

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