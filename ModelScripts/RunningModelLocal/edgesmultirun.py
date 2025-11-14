import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# -------------------------
# GPU setup
# -------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[INFO] Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print("[WARNING] GPU memory growth could not be set:", e)
else:
    print("[INFO] No GPU detected, using CPU")

# -------------------------
# CSV loading
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
# Image loading + preprocessing function
# -------------------------
def load_and_preprocess(img_path, target_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale 0-1
    img = tf.image.resize(img, target_size, preserve_aspect_ratio=True)
    
    # Pad to exact target size
    target_h, target_w = target_size
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    img = tf.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.0)
    
    return img

# -------------------------
# Main function
# -------------------------
def main(model_path, model_type, image_folder, batch_size=8):

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load JSON info
    model_dir = os.path.dirname(model_path)
    json_files = [f for f in os.listdir(model_dir) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError("No JSON file found in model directory.")
    json_path = os.path.join(model_dir, json_files[0])
    with open(json_path, "r") as f:
        model_info = json.load(f)

    param_mean = np.array(model_info.get("param_mean", [0.0, 0.0]))
    param_std = np.array(model_info.get("param_std", [1.0, 1.0]))
    image_size = tuple(model_info.get("image_size", [800, 800]))

    print(f"[INFO] Loaded JSON: image_size={image_size}, param_mean={param_mean}, param_std={param_std}")

    # Load CSVs
    input_csv_path = os.path.join(image_folder, "input_params.csv")
    output_csv_path = os.path.join(image_folder, "output_params.csv")
    raw_param_map = read_params_csv(input_csv_path)
    true_df = pd.read_csv(output_csv_path)
    true_map = {row["Image Name"]: row[model_type] for _, row in true_df.iterrows()}

    per_image_params = {
        name: (vals - param_mean) / param_std
        for name, vals in raw_param_map.items()
    }

    # -------------------------
    # Build tf.data.Dataset
    # -------------------------
    edges_folder = os.path.join(image_folder, "Edges")
    img_files = sorted([f for f in os.listdir(edges_folder) if f.lower().endswith(".png")])
    img_paths = [os.path.join(edges_folder, f) for f in img_files]

    param_list = []
    for f in img_files:
        p = per_image_params.get(f, (np.zeros_like(param_mean)-param_mean)/param_std)
        param_list.append(p)
    param_list = np.array(param_list, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, param_list))
    
    def _load_img_and_params(img_path, params):
        img = load_and_preprocess(img_path, target_size=image_size)
        return (img, params)
    
    dataset = dataset.map(_load_img_and_params, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # -------------------------
    # Run predictions
    # -------------------------
    results = []
    print(f"[INFO] Running predictions on {len(img_files)} images...")
    for batch_imgs, batch_params in tqdm(dataset, desc="Predicting"):
        preds = model.predict([batch_imgs, batch_params], verbose=0)
        for i in range(len(preds)):
            img_name = img_files[len(results)]
            true_value = true_map.get(img_name, None)
            pred_value = float(preds[i][0])
            delta = None if true_value is None else abs(pred_value - true_value)
            results.append({
                "Image Name": img_name,
                "True Value": true_value,
                "Predicted Value": pred_value,
                "Delta": delta
            })

    # -------------------------
    # Save results
    # -------------------------
    out_csv = os.path.join(image_folder, f"{model_type.replace(' ','_').replace('/','_')}_Predictions.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[INFO] Saved predictions to {out_csv}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CNN model on folder of images")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["Area (cm^2)", "Surface Tension (mN/m)", "Volume (ul)", "Curvature (1/cm)"])
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.model_path, args.model_type, args.image_folder, batch_size=args.batch_size)
