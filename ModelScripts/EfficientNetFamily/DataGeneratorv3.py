import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class ADSADataPipeline:
    def __init__(self, dataset_path, split='train', image_size=(512, 640),
                 output_type='Surface Tension (mN/m)', batch_size=32,
                 shuffle=True, random_state=42):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "Edges")
        self.input_csv = os.path.join(dataset_path, "input_params.csv")
        self.output_csv = os.path.join(dataset_path, "output_params.csv")
        self.split = split
        self.image_size = image_size
        self.output_type = output_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state

        # Load CSVs
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        # Merge on unique image name
        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))

        # Generate full image paths
        merged['path'] = merged['Image Name'].apply(lambda x: os.path.join(self.image_dir, x))
        merged = merged[merged['path'].apply(os.path.exists)]

        # Optional sanity check
        if len(merged) == 0:
            raise ValueError("No valid images found after merging CSVs with Edges folder.")

        # Split indices for train/val/test
        indices = np.arange(len(merged))
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

        if split == 'train':
            selected = train_idx
        elif split == 'val':
            selected = val_idx
        elif split == 'test':
            selected = test_idx
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.data = merged.iloc[selected].reset_index(drop=True)

        # Compute train normalization stats
        stats_path = os.path.join(dataset_path, "param_stats.npz")
        if split == 'train':
            self.param_mean = self.data[['Delta Rho (g/ml)', 'Scale Factor (cm/pixel)']].mean().values
            self.param_std  = self.data[['Delta Rho (g/ml)', 'Scale Factor (cm/pixel)']].std().values
            np.savez(stats_path, mean=self.param_mean, std=self.param_std)
        else:
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std  = stats["std"]

        # Store arrays for TensorFlow
        self.image_paths = self.data['path'].values
        self.params = self.data[['Delta Rho (g/ml)', 'Scale Factor (cm/pixel)']].values.astype(np.float32)
        self.outputs = self.data[self.output_type].values.astype(np.float32)

        # Augmentation layer (for training only)
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='nearest'
        )

    def _parse_function(self, path, param, y):
        # Load and decode image as 3 channels
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Original size
        original_h = tf.cast(tf.shape(image)[0], tf.float32)
        original_w = tf.cast(tf.shape(image)[1], tf.float32)

        # Resize with padding
        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])

        # Scale Factor adjustment
        scale_h = tf.cast(self.image_size[0], tf.float32) / original_h
        scale_w = tf.cast(self.image_size[1], tf.float32) / original_w
        scale = tf.minimum(scale_h, scale_w)
        param = tf.concat([param[:1], param[1:2] / scale], axis=0)

        # Normalize
        param = (param - self.param_mean) / self.param_std

        # Augmentation for training
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
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.image_paths), seed=self.random_state)
        ds = ds.map(lambda p, prm, y: self._parse_function(p, prm, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
