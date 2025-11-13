import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DataGeneratorv3 import ADSADataPipeline # your custom generator


# === SETTINGS ===
dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
output_type = "Surface Tension (mN/m)"
image_size = (1280, 1024)
num_samples = 20
split = "train"  # or 'val' or 'test'

# Load pipeline
pipeline = ADSADataPipeline(dataset_path, split=split, image_size=image_size, output_type=output_type, batch_size=1, shuffle=False)

print(f"Dataset size ({split}): {len(pipeline.image_paths)} images")

# Randomly sample indices
indices = random.sample(range(len(pipeline.image_paths)), min(num_samples, len(pipeline.image_paths)))

for idx in indices:
    img_path = pipeline.image_paths[idx]
    param = pipeline.params[idx]
    output = pipeline.outputs[idx]

    print("="*50)
    print(f"Image: {img_path}")
    
    # Original parameter before normalization (recompute)
    orig_param_df = pd.read_csv(os.path.join(dataset_path, "input_params.csv"))
    row = orig_param_df[orig_param_df["Image Name"] == os.path.basename(img_path)]
    if not row.empty:
        orig_param = row[['Delta Rho (g/ml)', 'Scale Factor (cm/pixel)']].values[0]
        print(f"Original parameters: {orig_param}")
    print(f"Normalized parameters: {param}")
    print(f"Output: {output}")

    # Optional: display image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    plt.imshow(img.numpy())
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()
