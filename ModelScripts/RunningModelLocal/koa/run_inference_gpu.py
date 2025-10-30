#!/usr/bin/env python3
"""
Run inference on a trained CNN model for ADSA data.

Example usage:
    python run_inference_gpu.py \
        --model_path /home/jordanw7/koa_scratch/ADSA-AI/Models/SurfaceTension_Model.keras \
        --model_type "Surface Tension" \
        --dataset_path /home/jordanw7/koa_scratch/ADSA-AI/Dataset/DataSetCombined \
        --batch_size 4
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import os
import time
from tensorflow.keras.models import load_model
from DataGeneratorv3 import ADSADataPipeline  # Ensure this file is in your PYTHONPATH

def main(model_path, model_type, dataset_path, batch_size, image_size=(800,800)):
    # model_path is passed as argument
    model_dir = os.path.dirname(model_path)
    json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]

    if not json_files:
        raise FileNotFoundError(f"No JSON file found in model directory: {model_dir}")

    # assuming there's only one stats JSON
    stats_path = os.path.join(model_dir, json_files[0])
    with open(stats_path, "r") as f:
        stats = json.load(f)

    param_mean = np.array(stats["param_mean"], dtype=np.float32)
    param_std = np.array(stats["param_std"], dtype=np.float32)
    image_size = tuple(stats["image_size"])
    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    print(f"[INFO] Preparing test data from: {dataset_path}")
    test_pipeline = ADSADataPipeline(
        dataset_path,
        split='test',
        image_size=image_size,
        output_type=model_type,
        batch_size=batch_size
    )
    test_gen = test_pipeline.get_dataset()
    image_paths = test_pipeline.image_paths

    all_true, all_pred, all_times, all_names = [], [], [], []
    start_total = time.time()

    print("[INFO] Starting inference...")
    for batch_idx, ((X_batch, params_batch), y_batch) in enumerate(test_gen):
        start_batch = time.time()
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed_batch = time.time() - start_batch

        all_true.extend(y_batch)
        all_pred.extend(preds_batch.flatten())
        all_times.extend([elapsed_batch / len(y_batch)] * len(y_batch))

        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(y_batch)
        all_names.extend(image_paths[start_idx:end_idx])

    end_total = time.time()
    total_elapsed_time = end_total - start_total
    print(f"[INFO] Inference completed in {total_elapsed_time:.2f} seconds")

    # Save results
    model_dir = os.path.dirname(model_path)
    output_csv = os.path.join(model_dir, f"results_{model_type.replace(' ', '_')}.csv")
    
    results_df = pd.DataFrame({
        "image_name": all_names,
        "True_Value": all_true,
        "Predicted_Value": all_pred,
        "Prediction_Time_s": all_times
    })
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to {output_csv}")

    # Optional plot
    plt.figure(figsize=(6,6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    if len(all_true) > 1:
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
    plt.xlabel(f"True {model_type}")
    plt.ylabel(f"Predicted {model_type}")
    plt.title(f"Predicted vs True {model_type}")
    plt.grid(True)
    plt.tight_layout()
    plot_name = os.path.join(model_dir, f"pred_vs_true_{model_type.replace(' ', '_')}.png")
    plt.savefig(plot_name)
    print(f"[INFO] Plot saved to {plot_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained CNN model on ADSA dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.keras)")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["Surface Area", "Surface Tension (mN/m)", "Volume", "Curvature"],
                        help="Which type of output the model predicts")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 640], help="Input image size (H W)")
    args = parser.parse_args()

    main(args.model_path, args.model_type, args.dataset_path, args.batch_size, tuple(args.image_size))