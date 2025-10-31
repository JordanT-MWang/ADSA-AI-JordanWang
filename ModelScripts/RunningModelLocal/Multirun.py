import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import argparse

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_image(img_path, target_size=(512, 640)):

    """Resize + pad + convert to 3 channels."""
    img = Image.open(img_path).convert("L")
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

    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# -------------------------
# Read params.txt
# -------------------------
def read_params(param_file):
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
# Main function
# -------------------------
def main(model_path, model_type, image_folder):
    # Load model
    model = load_model(model_path, compile=False)

    # Set normalization stats (replace with your training values)
    param_mean = np.array([9.9271584e-01, 6.0119742e-04])
    param_std  = np.array([0.00061441, 0.00056276])

    # Read parameters from params.txt
    param_file = os.path.join(image_folder, "params.txt")
    params = read_params(param_file)
    params = (params - param_mean) / param_std
    params = np.expand_dims(params, axis=0)

    # Process images
    edges_folder = os.path.join(image_folder, "Edges")
    if not os.path.exists(edges_folder):
        raise FileNotFoundError(f"Edges folder not found: {edges_folder}")

    img_files = [f for f in os.listdir(edges_folder) if f.lower().endswith(".png")]
    if not img_files:
        raise FileNotFoundError("No PNG images found in Edges folder")
    img_files.sort()  # optional alphabetical sort

    predictions = []
    for img_name in img_files:
        img_path = os.path.join(edges_folder, img_name)
        img = preprocess_image(img_path)
        pred = model.predict([img, params], verbose=0)
        predictions.append(pred[0][0])

    # Save .dat file
    dat_filename = f"{model_type.replace(' ', '_')}_Predictions.dat"
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
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing Edges/ and params.txt")
    args = parser.parse_args()

    main(args.model_path, args.model_type, args.image_folder)
