import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2

class ADSADataPipeline:
    def __init__(self, dataset_path, split='train', image_size=(512, 640),
                 output_type='Surface Tension', batch_size=32, shuffle=True,
                 normalize_output=False):

        # -----------------------
        # Paths & Config
        # -----------------------
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "Edges")
        self.input_csv = os.path.join(dataset_path, "input_params.csv")
        self.output_csv = os.path.join(dataset_path, "output_params.csv")
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_type = output_type
        self.split = split
        self.shuffle = shuffle
        self.normalize_output = normalize_output
        
        # -----------------------
        # Load main tables
        # -----------------------
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)

        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))

        # Filter for existing images only
        valid = merged["Image Name"].apply(lambda nm: 
                                           os.path.exists(os.path.join(self.image_dir, nm)))
        merged = merged[valid]

        # -----------------------
        # Extract data
        # -----------------------
        self.image_paths = merged["Image Name"].apply(
            lambda nm: os.path.join(self.image_dir, nm)).values
        
        self.params = merged[["Delta Rho (g/ml)",
                              "Scale Factor (cm/pixel)"]].astype(np.float32).values
        
        self.outputs = merged[self.output_type].astype(np.float32).values

        # -----------------------
        # Make splits ONCE
        # -----------------------
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(self.image_paths))

        train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        if split == "train": sel = train_idx
        elif split == "val": sel = val_idx
        else: sel = test_idx

        self.image_paths = self.image_paths[sel]
        self.params = self.params[sel]
        self.outputs = self.outputs[sel]

        # -----------------------
        # Load or compute normalization
        # -----------------------
        stats_path = os.path.join(dataset_path, "global_stats.npz")

        if split == "train":
            self.param_mean = self.params.mean(axis=0)
            self.param_std = self.params.std(axis=0) + 1e-8   # avoid zero-div

            if self.normalize_output:
                self.output_mean = self.outputs.mean()
                self.output_std = self.outputs.std() + 1e-8
            else:
                self.output_mean = 0.0
                self.output_std = 1.0

            np.savez(stats_path,
                     p_mean=self.param_mean,
                     p_std=self.param_std,
                     o_mean=self.output_mean,
                     o_std=self.output_std)

        else:
            stats = np.load(stats_path)
            self.param_mean = stats["p_mean"]
            self.param_std = stats["p_std"]
            self.output_mean = stats["o_mean"]
            self.output_std = stats["o_std"]

        # Augmentation layer for translation
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='nearest'
        )

    # -----------------------------------------------------------
    # Core preprocessing per sample
    # -----------------------------------------------------------
    def _parse_function(self, path, param, y):
        # Load PNG grayscale
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Get original size
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)

        # Resize-with-pad scaling correction
        scale_h = self.image_size[0] / h
        scale_w = self.image_size[1] / w
        scale = tf.minimum(scale_h, scale_w)

        # Apply resize-with-pad
        image = tf.image.resize_with_pad(image,
                                         self.image_size[0],
                                         self.image_size[1])

        # Correct param scale factor since image changed
        delta_rho = param[0]
        scale_factor = param[1] / scale
        param = tf.stack([delta_rho, scale_factor])

        # Normalize parameters
        param = (param - self.param_mean) / self.param_std

        # Normalize output (optional)
        y = (y - self.output_mean) / self.output_std

        # Augmentation only in train
        if self.split == "train":
            image = self._augment(image)

        return (image, param), y

    # -----------------------------------------------------------
    # Light augmentations
    # -----------------------------------------------------------
    def _augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = self.translation_layer(image[None, ...])[0]
        return image

    # -----------------------------------------------------------
    # Build dataset
    # -----------------------------------------------------------
    def get_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.params, self.outputs))

        if self.shuffle and self.split == "train":
            ds = ds.shuffle(20000)

        ds = ds.map(self._parse_function,
                    num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
