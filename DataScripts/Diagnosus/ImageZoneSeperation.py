import os
import pandas as pd
import shutil

# === CONFIGURATION ===
base_dir = r"E:\Data\Jordan\Jordan Wang AI-ADSA\Dataset\DataSetCombined"
analyzed_csv = r"E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\EfficientNetFamily\SurfaceTension\10v\ST_Model_Predictions_Analyzed_etf_v10.csv"
output_category_dirname = "SurfaceTension"  # main output directory name

# === DERIVED PATHS ===
edges_dir = os.path.join(base_dir, "Edges")
input_params_path = os.path.join(base_dir, "input_params.csv")
output_params_path = os.path.join(base_dir, "output_params.csv")
output_root = os.path.join(base_dir, output_category_dirname)

# Create the top-level category directory if it doesn’t exist
os.makedirs(output_root, exist_ok=True)

# === READ FILES ===
print("[INFO] Loading CSVs...")
df_analyzed = pd.read_csv(analyzed_csv)
df_input = pd.read_csv(input_params_path)
df_output = pd.read_csv(output_params_path)

# Clean up the image name (remove full path)
df_analyzed["Image Name"] = df_analyzed["image_name"].apply(lambda x: os.path.basename(x))

# Normalize zone labels
df_analyzed["Zone"] = df_analyzed["Zone"].str.lower().str.strip()

# === PROCESS EACH ZONE ===
zones = ["black", "orange", "red"]

for zone in zones:
    print(f"\n[INFO] Processing zone: {zone}")
    zone_dir = os.path.join(output_root, f"{zone}_edges")
    zone_edges_dir = os.path.join(zone_dir, "Edges")

    # Create folders
    os.makedirs(zone_edges_dir, exist_ok=True)
    
    # Filter images for this zone
    zone_df = df_analyzed[df_analyzed["Zone"] == zone]
    if zone_df.empty:
        print(f"[WARN] No images found for zone: {zone}")
        continue
    
    image_names = zone_df["Image Name"].tolist()
    
    # Filter input/output params
    zone_input_df = df_input[df_input["Image Name"].isin(image_names)]
    zone_output_df = df_output[df_output["Image Name"].isin(image_names)]
    
    # Save filtered CSVs
    zone_input_csv = os.path.join(zone_dir, "input_params.csv")
    zone_output_csv = os.path.join(zone_dir, "output_params.csv")
    zone_input_df.to_csv(zone_input_csv, index=False)
    zone_output_df.to_csv(zone_output_csv, index=False)
    
    # Copy image files into Edges/
    print(f"[INFO] Copying {len(image_names)} images to {zone_edges_dir}")
    for img_name in image_names:
        src_path = os.path.join(edges_dir, img_name)
        dst_path = os.path.join(zone_edges_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"[WARN] Missing image file: {src_path}")

print(f"\n[DONE] Categorized data saved under: {output_root}")
