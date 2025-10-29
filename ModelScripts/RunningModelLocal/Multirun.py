import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os
import argparse

# -------------------------
# Functions
"""use example python run_folder_model.py ^
    --model_path "E:/Models/SurfaceTension_Model_Large_Mobile_V1.keras" ^
    --model_type "Surface Tension" ^
    --image_folder "E:/Data/Jordan/Jordan Wang AI-ADSA/Surface tension - AI/Set 1 images"
"""
# -------------------------
# Retrieve normalization stats if saved in model metadata
param_mean = np.array([9.9271584e-01, 6.0119742e-04])
param_std = np.array([0.00061441, 0.00056276])
def read_params(param_file):
    """Read params.txt with Scale Factor and Density."""
    scale = None
    density = None
    with open(param_file, "r") as f:
        for line in f:
            if "Scale Factor" in line:
                scale = float(line.split(":")[1].strip())
            elif "Density" in line:
                density = float(line.split(":")[1].strip().split()[0])
    if scale is None or density is None:
        raise ValueError("Could not parse params.txt")
    return np.array([density, scale], dtype=np.float32)

def preprocess_image(img_path, target_size=(512, 640)):
    """Resize + pad + 3-channel conversion."""
    img = Image.open(img_path).convert("L")
    img = np.array(img, dtype=np.float32)
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
    img /= 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# -------------------------
# Main
# -------------------------
def main(model_path, model_type, image_folder):
    # Load model
    model = load_model(model_path, compile=False)

    # Read params.txt
    param_file = os.path.join(image_folder, "params.txt")
    params = read_params(param_file)
    params_batch = np.expand_dims(params, axis=0)

    # Find all PNG images in Edges folder
    edges_folder = os.path.join(image_folder, "Edges")
    if not os.path.exists(edges_folder):
        raise FileNotFoundError(f"Edges folder not found: {edges_folder}")
    
    img_files = [f for f in os.listdir(edges_folder) if f.lower().endswith(".png")]
    if not img_files:
        raise FileNotFoundError("No PNG images found in Edges folder")

    predictions = []
    for img_name in img_files:
        img_path = os.path.join(edges_folder, img_name)
        img = preprocess_image(img_path)
        params = np.array(params, dtype=np.float32)
        params = (params - param_mean) / param_std  # normalize
        params = np.expand_dims(params, axis=0)
        pred = model.predict([img, params_batch], verbose=0)
        predictions.append(pred[0][0])

    # Save as .dat file
    dat_filename = f"{model_type.replace(' ', '_')}_Predictions.dat"
    dat_path = os.path.join(image_folder, dat_filename)
    np.savetxt(dat_path, predictions, fmt="%.6f")
    print(f"[INFO] Predictions saved to {dat_path}")

# -------------------------
# CLI Arguments
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN on a folder of images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.keras)")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["Surface Area", "Surface Tension", "Volume", "Curvature"])
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing Edges/ and params.txt")

    args = parser.parse_args()
    main(args.model_path, args.model_type, args.image_folder)