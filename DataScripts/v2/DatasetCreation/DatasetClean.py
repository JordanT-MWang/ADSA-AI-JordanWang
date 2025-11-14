import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm


def read_data_dat(path):
    """
    Reads data.dat and returns a dataframe with actual data lines only.
    Skips header lines and blank lines.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                # first column must be int (n)
                n = int(parts[0])
                vals = list(map(float, parts[1:]))
                rows.append([n] + vals)
            except:
                continue
    
    df = pd.DataFrame(rows, columns=[
        "n", "Time", "SurT", "Curv", "Area", "Volu", "CAgl"
    ])
    return df


def clean_run(run_path):
    edges_dir = os.path.join(run_path, "Edges")
    input_csv = os.path.join(run_path, "input_params.csv")
    output_csv = os.path.join(run_path, "output_params.csv")
    data_dat = os.path.join(run_path, "data.dat")

    if not os.path.exists(edges_dir) or not os.path.exists(input_csv) \
       or not os.path.exists(output_csv) or not os.path.exists(data_dat):
        print(f"Skipping {run_path} — missing required files")
        return

    print(f"\n===============================")
    print(f"CLEANING RUN: {run_path}")
    print("===============================")

    df_in = pd.read_csv(input_csv)
    df_out = pd.read_csv(output_csv)
    df_dat = read_data_dat(data_dat)

    print(f"Images in Edges/      = {len(os.listdir(edges_dir))}")
    print(f"Rows in input.csv     = {len(df_in)}")
    print(f"Rows in output.csv    = {len(df_out)}")
    print(f"Rows in data.dat      = {len(df_dat)}")

    # -----------------------------------------------
    # Step 1 — Detect corrupted PNGs
    # -----------------------------------------------
    corrupted = []
    for img in os.listdir(edges_dir):
        if not img.endswith(".png"):
            continue
        p = os.path.join(edges_dir, img)
        try:
            with Image.open(p) as im:
                im.verify()
        except:
            corrupted.append(img)

    if corrupted:
        print(f"❌ Found {len(corrupted)} corrupted PNGs — moving to run folder")
        for img in corrupted:
            shutil.move(os.path.join(edges_dir, img), os.path.join(run_path, img))

    # Filter them out of input.csv
    df_in = df_in[~df_in["Image Name"].isin(corrupted)]
    df_out = df_out[~df_out["Image Name"].isin(corrupted)]

    # -----------------------------------------------
    # Step 2 — Remove extra images not in output file
    # -----------------------------------------------
    valid_imgs = set(df_out["Image Name"].values)
    all_imgs = set([f for f in os.listdir(edges_dir) if f.endswith(".png")])

    extra_imgs = all_imgs - valid_imgs

    if extra_imgs:
        print(f"⚠️ Found {len(extra_imgs)} extra images not listed in output.csv")
        for img in extra_imgs:
            src = os.path.join(edges_dir, img)
            dest = os.path.join(run_path, img)
            shutil.move(src, dest)

        # Remove from input_params too
        df_in = df_in[~df_in["Image Name"].isin(extra_imgs)]

    # -----------------------------------------------
    # Step 3 — Align rows to data.dat
    # -----------------------------------------------
    needed_rows = len(df_dat)
    have_rows = len(df_out)

    if have_rows != needed_rows:
        print(f"⚠️ Mismatch: output has {have_rows} rows but data.dat has {needed_rows}")

    # Truncate OR raise if inconsistent
    min_len = min(needed_rows, have_rows)

    df_out = df_out.iloc[:min_len].reset_index(drop=True)
    df_in = df_in.iloc[:min_len].reset_index(drop=True)

    # -----------------------------------------------
    # Step 4 — Save cleaned CSVs
    # -----------------------------------------------
    df_in.to_csv(input_csv, index=False)
    df_out.to_csv(output_csv, index=False)

    print(f"✔ Cleaned run: {run_path}")
    print(f"Final aligned rows = {len(df_out)}")


def clean_parent(parent_dir):
    for run in sorted(os.listdir(parent_dir)):
        run_path = os.path.join(parent_dir, run)
        if os.path.isdir(run_path):
            clean_run(run_path)


if __name__ == "__main__":
    parent = input("Enter parent directory path: ").strip()
    clean_parent(parent)
    print("\n✨ Cleaning complete!")
