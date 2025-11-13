import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2

class ADSADataPipeline:
    def __init__(self, dataset_path, split='train', image_size=(512, 640),
                 output_type='Surface Tension', batch_size=32, shuffle=True, random_state=42):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "Edges")
        self.input_csv = os.path.join(dataset_path, "input_params.csv")
        self.output_csv = os.path.join(dataset_path, "output_params.csv")
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_type = output_type
        self.split = split
        self.shuffle = shuffle
        self.random_state = random_state

        # Translation layer (same as before)
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='nearest'
        )

        # --- Alignment-safe CSV merge ---
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        # Strip spaces & ensure consistent naming
        input_df["Image Name"] = input_df["Image Name"].str.strip()
        output_df["Image Name"] = output_df["Image Name"].str.strip()

        # Inner join ensures only matching images are kept
        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))

        image_paths, params, outputs = [], [], []
        for _, row in merged.iterrows():
            path = os.path.join(self.image_dir, row["Image Name"])
            if os.path.exists(path):
                image_paths.append(path)
                params.append([row["Delta Rho (g/ml)"], row["Scale Factor (cm/pixel)"]])
                outputs.append(row[self.output_type])

        image_paths = np.array(image_paths)
        params = np.array(params, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)

        # --- Dataset splitting ---
        from sklearn.model_selection import train_test_split
        train_idx, temp_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

        if split == 'train':
            idx = train_idx
        elif split == 'val':
            idx = val_idx
        else:
            idx = test_idx

        self.image_paths = image_paths[idx]
        self.params = params[idx]
        self.outputs = outputs[idx]

        # --- Parameter normalization ---
        stats_path = os.path.join(dataset_path, "param_stats.npz")
        if split == 'train':
            self.param_mean = np.mean(self.params, axis=0)
            self.param_std = np.std(self.params, axis=0)
            np.savez(stats_path, mean=self.param_mean, std=self.param_std)
        else:
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

    def _parse_function(self, path, param, y):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize with pad + maintain scale adjustment
        original_h = tf.cast(tf.shape(image)[0], tf.float32)
        original_w = tf.cast(tf.shape(image)[1], tf.float32)
        scale_h = tf.cast(self.image_size[0], tf.float32) / original_h
        scale_w = tf.cast(self.image_size[1], tf.float32) / original_w
        scale = tf.minimum(scale_h, scale_w)

        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])

        # Adjust scale factor
        param = tf.concat([param[:1], param[1:2] / scale], axis=0)

        # Normalize params
        param = (param - self.
