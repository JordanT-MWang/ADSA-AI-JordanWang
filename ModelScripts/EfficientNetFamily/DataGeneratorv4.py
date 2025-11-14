import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class ADSADataPipeline:
    def __init__(self, dataset_path, split='train', image_size=(512, 640),
                 output_type='Surface Tension', batch_size=32, shuffle=True,
                 random_state=42, cache=True):
        """
        dataset_path: folder containing Edges/, input_params.csv, output_params.csv
        split: 'train', 'val', or 'test'
        output_type: column name in output CSV to predict
        """
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "Edges")
        self.input_csv = os.path.join(dataset_path, "input_params.csv")
        self.output_csv = os.path.join(dataset_path, "output_params.csv")
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_type = output_type
        self.split = split
        self.shuffle = shuffle
        self.cache = cache
        self.random_state = random_state

        # TensorFlow augmentation layer
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='nearest'
        )

        # Load CSVs
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        # Merge on image name
        merged = pd.merge(input_df, output_df, on="Image Name")
        # Filter out missing images
        merged["path"] = merged["Image Name"].apply(lambda x: os.path.join(self.image_dir, x))
        merged = merged[merged["path"].apply(os.path.exists)]
        # Sort by filename to guarantee alignment
        merged = merged.sort_values("Image Name").reset_index(drop=True)

        # Extract data arrays
        image_paths = merged["path"].to_numpy()
        params = merged[["Delta Rho (g/ml)", "Scale Factor (cm/pixel)"]].to_numpy(dtype=np.float32)
        outputs = merged[self.output_type].to_numpy(dtype=np.float32)

        # Train/val/test split
        train_idx, temp_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2,
                                              random_state=self.random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=self.random_state)

        if split == 'train': idx = train_idx
        elif split == 'val': idx = val_idx
        else: idx = test_idx

        self.image_paths = image_paths[idx]
        self.params = params[idx]
        self.outputs = outputs[idx]

        # Compute normalization on train split
        if split == 'train':
            self.param_mean = np.mean(self.params, axis=0)
            self.param_std = np.std(self.params, axis=0) + 1e-7
            # Save stats for reuse
            np.savez(os.path.join(dataset_path, "param_stats.npz"),
                     mean=self.param_mean, std=self.param_std)
        else:
            stats_path = os.path.join(dataset_path, "param_stats.npz")
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

    def _parse_function(self, path, param, y):
        # Read and decode image
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize with padding
        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])

        # Adjust scale factor
        original_h = tf.cast(tf.shape(image)[0], tf.float32)
        original_w = tf.cast(tf.shape(image)[1], tf.float32)
        scale_h = tf.cast(self.image_size[0], tf.float32) / original_h
        scale_w = tf.cast(self.image_size[1], tf.float32) / original_w
        scale = tf.minimum(scale_h, scale_w)
        param = tf.concat([param[:1], param[1:2] / scale], axis=0)

        # Normalize params
        param = (param - self.param_mean) / self.param_std

        # Training augmentations
        if self.split == 'train':
            image = self._augment(image)

        return (image, param), y

    def _augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = self.translation_layer(tf.expand_dims(image, 0))[0]
        return image

    def get_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.params, self.outputs))
        if self.shuffle and self.split == 'train':
            ds = ds.shuffle(buffer_size=len(self.image_paths))
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            ds = ds.cache()  # <--- caching for speed
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
