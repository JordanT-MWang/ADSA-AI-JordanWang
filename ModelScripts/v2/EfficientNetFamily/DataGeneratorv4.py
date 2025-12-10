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
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15,
            fill_mode='nearest'
        )
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))
        image_paths, params, outputs = [], [], []

        for _, row in merged.iterrows():
            path = os.path.join(self.image_dir, row["Image Name"])
            if os.path.exists(path):
                image_paths.append(path)
                params.append([row["Scale Factor (cm/pixel)"]])
                outputs.append(row[self.output_type])

        # Split (80/10/10)
        from sklearn.model_selection import train_test_split
        train_idx, temp_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

        if split == 'train': idx = train_idx
        elif split == 'val': idx = val_idx
        else: idx = test_idx

        self.image_paths = np.array(image_paths)[idx]
        self.params = np.array(params, dtype=np.float32)[idx]
        self.outputs = np.array(outputs, dtype=np.float32)[idx]

        # Compute normalization on train split
        if split == 'train':
            self.param_mean = np.mean(self.params, axis=0)
            self.param_std = np.std(self.params, axis=0)
        else:
            # Load saved stats from train split
            stats_path = os.path.join(dataset_path, "param_stats.npz")
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

        # Save train stats for others
        if split == 'train':
            np.savez(os.path.join(dataset_path, "param_stats.npz"),
                     mean=self.param_mean, std=self.param_std)

    def _parse_function(self, path, param, y):

        # 1. Load raw image (uint8)
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)

        original_h = tf.cast(tf.shape(image)[0], tf.float32)
        original_w = tf.cast(tf.shape(image)[1], tf.float32)

        # 2. Compute scale used by resize_with_pad
        target_h, target_w = self.image_size
        scale_h = target_h / original_h
        scale_w = target_w / original_w
        resize_scale = tf.minimum(scale_h, scale_w)

        # 3. Resize + pad (still uint8 at this point)
        image = tf.image.resize_with_pad(image, target_h, target_w)

        # 4. Apply augmentations BEFORE float normalization
        if self.split == 'train':
            image = self._augment(image)

        # 5. Convert to float AFTER all geometry transforms
        image = tf.cast(image, tf.float32) / 255.0

        # 6. Fix scale factor correctly
        # param[0] = cm/pixel_original
        # new cm/pixel = old / resize_scale
        corrected_param = param[0] / resize_scale

        # 7. Normalize parameters
        corrected_param = corrected_param self.param_mean
        corrected_param = tf.expand_dims(corrected_param, 0)

        return (image, corrected_param), y

    def _augment(self, image):
        # Lightweight TensorFlow augmentations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        # Use built-in layer for translation
        image = self.translation_layer(tf.expand_dims(image, 0))[0]

        return image

    def get_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.params, self.outputs))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.image_paths))
        ds = ds.map(lambda p, prm, y: self._parse_function(p, prm, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
