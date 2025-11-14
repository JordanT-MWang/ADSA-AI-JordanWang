import tensorflow as tf
import numpy as np
import os
import pandas as pd

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

        # Translation augmentation layer
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='nearest'
        )

        # -----------------------------
        # Load input/output CSVs
        # -----------------------------
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        # Merge on Image Name (NO SPLIT column required)
        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))

        image_paths, params, outputs = [], [], []

        # Collect aligned rows
        for _, row in merged.iterrows():
            path = os.path.join(self.image_dir, row["Image Name"])
            if os.path.exists(path):
                image_paths.append(path)
                params.append([row["Delta Rho (g/ml)"], row["Scale Factor (cm/pixel)"]])
                outputs.append(row[self.output_type])

        # -----------------------------
        # Train/Val/Test split internally
        # -----------------------------
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(image_paths))

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=random_state
        )

        if split == 'train': idx = train_idx
        elif split == 'val': idx = val_idx
        else: idx = test_idx

        # Keep only selected rows
        self.image_paths = np.array(image_paths)[idx]
        self.params = np.array(params, dtype=np.float32)[idx]
        self.outputs = np.array(outputs, dtype=np.float32)[idx]

        # -----------------------------
        # Compute or load normalization
        # -----------------------------
        stats_path = os.path.join(dataset_path, "param_stats.npz")
        if split == 'train':
            self.param_mean = self.params.mean(axis=0)
            self.param_std = self.params.std(axis=0) + 1e-7
            np.savez(stats_path, mean=self.param_mean, std=self.param_std)
        else:
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

    # -----------------------------
    # Image loader + param adjustment
    # -----------------------------
    def _parse_function(self, path, param, y):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        orig_h = tf.cast(tf.shape(image)[0], tf.float32)
        orig_w = tf.cast(tf.shape(image)[1], tf.float32)

        scale_h = tf.cast(self.image_size[0], tf.float32) / orig_h
        scale_w = tf.cast(self.image_size[1], tf.float32) / orig_w
        scale = tf.minimum(scale_h, scale_w)

        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
        param = tf.concat([param[:1], param[1:2] / scale], axis=0)
        param = (param - self.param_mean) / self.param_std

        if self.split == 'train':
            image = self._augment(image)

        return (image, param), y

    # -----------------------------
    # Augmentation
    # -----------------------------
    def _augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = self.translation_layer(tf.expand_dims(image, 0))[0]
        return image

    # -----------------------------
    # Create tf.data.Dataset with caching
    # -----------------------------
    def get_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.params, self.outputs))

        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.image_paths))

        # Training: no caching (augmentations must run every epoch)
        # Validation/Test: cache fully in RAM
        if self.split != 'train':
            ds = ds.cache()  # RAM caching

        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
