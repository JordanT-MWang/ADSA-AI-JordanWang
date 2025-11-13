import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import argparse
import json
import pandas as pd

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_image(img_path, target_size=(512, 640)):
    """Resize + pad + convert to 3 channels."""
    #for grey
    #img = Image.open(img_path).convert("L")
    #for rgb
    img = Image.open(img_path).convert("RGB")
    img = img_to_array(img)

    target_h, target_w = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=0)
    img = img / 255.0
    #for grey
    #img = np.expand_dims(img, axis=-1)
    #img = np.repeat(img, 3, axis=-1)
    #img = np.expand_dims(img, axis=0)
    #for rgb
    img = np.expand_dims(img, axis=0)  # shape (1, H, W, 3)
    return img.astype(np.float32)

# -------------------------
# Read params.txt
# -------------------------
def read_params_txt(param_file):
    scale_factor = None
    density = None
    with open(param_file, "r") as f:
        for line in f:
            if "Scale Factor" in line:
                scale_factor = float(line.split(":")[1].strip())
            elif "Density" in line:
                density = float(line.split(":")[1].strip().split()[0])
    if scale_factor is None or density is None:
        raise ValueError("Could not read params.txt")
    return np.array([density, scale_factor], dtype=np.float32)

# -------------------------
# Read input_params.csv
# -------------------------
def read_params_csv(csv_path):
    df = pd.read_csv(csv_path)
    param_map = {}
    for _, row in df.iterrows():
        image_name = row["Image Name"]
        delta_rho = float(row["Delta Rho (g/ml)"])
        scale_factor = float(row["Scale Factor (cm/pixel)"])
        param_map[image_name] = np.array([delta_rho, scale_factor], dtype=np.float32)
    return param_map

# -------------------------
# Main function
# -------------------------
def main(model_path, model_type, image_folder):
    # Load model
    model = load_model(model_path, compile=False)

    # Automatically find JSON in model directory
    model_dir = os.path.dirname(model_path)
    json_files = [f for f in os.listdir(model_dir) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {model_dir} for model parameters")
    if len(json_files) > 1:
        print(f"⚠️  Multiple JSON files found in {model_dir}, using the first one: {json_files[0]}")
    json_path = os.path.join(model_dir, json_files[0])
    
    with open(json_path, "r") as f:
        model_info = json.load(f)

    param_mean = np.array(model_info.get("param_mean", [0.0, 0.0]))
    param_std = np.array(model_info.get("param_std", [1.0, 1.0]))
    image_size = tuple(model_info.get("image_size", [512, 640]))

    print(f"[INFO] Loaded model info from JSON: image_size={image_size}, param_mean={param_mean}, param_std={param_std}")

    # Determine parameters per image
    param_file = os.path.join(image_folder, "params.txt")
    input_csv_path = os.path.join(image_folder, "input_params.csv")
    per_image_params = {}
    if os.path.exists(param_file):
        global_params = read_params_txt(param_file)
        # Normalize
        global_params = (global_params - param_mean) / param_std
        print(f"[INFO] Using params.txt for normalization: {global_params}")
    elif os.path.exists(input_csv_path):
        raw_param_map = read_params_csv(input_csv_path)
        # Normalize per image
        for k, v in raw_param_map.items():
            per_image_params[k] = (v - param_mean) / param_std
        print(f"[INFO] Using input_params.csv for per-image normalization")
    else:
        raise FileNotFoundError("Neither params.txt nor input_params.csv found in the image folder")

    # Process images
    edges_folder = os.path.join(image_folder, "Edges")
    if not os.path.exists(edges_folder):
        raise FileNotFoundError(f"Edges folder not found: {edges_folder}")

    img_files = [f for f in os.listdir(edges_folder) if f.lower().endswith(".png")]
    if not img_files:
        raise FileNotFoundError("No PNG images found in Edges folder")
    img_files.sort()

    predictions = []
    for img_name in img_files:
        img_path = os.path.join(edges_folder, img_name)
        img = preprocess_image(img_path, target_size=image_size)

        # Get params
        if per_image_params:
            params = per_image_params.get(img_name)
            if params is None:
                print(f"⚠️  {img_name} not found in input_params.csv, using global mean")
                params = (np.array(param_mean) - param_mean) / param_std
        else:
            params = global_params

        params = np.expand_dims(params, axis=0)
        pred = model.predict([img, params], verbose=0)
        predictions.append(pred[0][0])

    # Save .dat file
    os.makedirs(image_folder, exist_ok=True)
    # Sanitize the model_type to remove characters that break Windows filenames
    safe_model_type = model_type.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    dat_filename = f"{safe_model_type}_Predictions.dat"
    dat_path = os.path.join(image_folder, dat_filename)


    np.savetxt(dat_path, predictions, fmt="%.6f")
    print(f"[INFO] Predictions saved to {dat_path}")

# -------------------------
# Argument parser
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN model on folder of images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.keras)")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["Area (cm^2)", "Surface Tension (mN/m)", "Volume (ul)", "Curvature (1/cm)"])
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing Edges/ and params.txt or input_params.csv")
    args = parser.parse_args()

    main(args.model_path, args.model_type, args.image_folder)
