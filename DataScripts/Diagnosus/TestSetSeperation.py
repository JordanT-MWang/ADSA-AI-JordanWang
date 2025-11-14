import os
import pandas as pd
import shutil

# === CONFIGURATION ===
base_dir = r"E:\Data\Jordan\Jordan Wang AI-ADSA\Dataset\DataSetCombined"
analyzed_csv = r"E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\EfficientNetFamily\SurfaceTension\10v\ST_Model_Predictions.csv"
output_category_dirname = "SurfaceTension_All"   # output folder containing all cleaned data

# === DERIVED PATHS ===
edges_dir = os.path.join(base_dir, "Edges")
input_params_path = os.path.join(base_dir, "input_params.csv")
output_params_path = os.path.join(base_dir, "output_params.csv")
output_root = os.path.join(base_dir, output_category_dirname)
output_edges_dir = os.path.join(output_root, "Edges")

# Create output directories
os.makedirs(output_edges_dir, exist_ok=True)

# === LOAD CSVs ===
print("[INFO] Loading CSVs...")
df_analyzed = pd.read_csv(analyzed_csv)
df_input = pd.read_csv(input_params_path)
df_output = pd.read_csv(output_params_path)

# Extract clean image names (remove folder paths)
df_analyzed["Image Name"] = df_analyzed["image_name"].apply(lambda x: os.path.basename(str(x)))
image_names = df_analyzed["Image Name"].unique().tolist()

print(f"[INFO] Found {len(image_names)} images in analyzed CSV")

# === FILTER INPUT & OUTPUT PARAMETER FILES ===
filtered_input_df = df_input[df_input["Image Name"].isin(image_names)]
filtered_output_df = df_output[df_output["Image Name"].isin(image_names)]

# Save new CSVs
filtered_input_df.to_csv(os.path.join(output_root, "input_params.csv"), index=False)
filtered_output_df.to_csv(os.path.join(output_root, "output_params.csv"), index=False)

print("[INFO] Saved filtered input/output parameter CSVs")

# === COPY IMAGES ===
copied = 0
missing = 0

print(f"[INFO] Copying images to: {output_edges_dir}")
for img_name in image_names:
    src = os.path.join(edges_dir, img_name)
    dst = os.path.join(output_edges_dir, img_name)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        print(f"[WARN] Missing image file: {src}")
        missing += 1

print(f"[DONE] Completed copy. Copied={copied}, Missing={missing}")
print(f"[DONE] Output saved to: {output_root}")
