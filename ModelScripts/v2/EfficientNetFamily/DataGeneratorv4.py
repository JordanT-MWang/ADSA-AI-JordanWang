import tensorflow as tf
import numpy as np
import os
import pandas as pd


# Disable XLA so tf.string in dataset will not crash
tf.config.optimizer.set_jit(False)


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

        # ------------------
        # Augmentations
        # ------------------
        self.translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15,
            fill_mode='nearest'
        )

        # ------------------
        # Load CSV files
        # ------------------
        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)        
        merged = pd.merge(input_df, output_df, on="Image Name")

        image_paths = []
        params = []
        outputs = []

        for _, row in merged.iterrows():
            img_path = os.path.join(self.image_dir, row["Image Name"])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                params.append([row["Scale Factor (cm/pixel)"]])
                outputs.append(row[self.output_type])

        # ------------------
        # Split train/val/test
        # ------------------
        from sklearn.model_selection import train_test_split
        idxs = np.arange(len(image_paths))
        train_idx, temp_idx = train_test_split(idxs, test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

        if split == 'train': idx = train_idx
        elif split == 'val': idx = val_idx
        else: idx = test_idx

        self.image_paths = np.array(image_paths)[idx]
        self.params = np.array(params, dtype=np.float32)[idx]
        self.outputs = np.array(outputs, dtype=np.float32)[idx]

        # ------------------
        # Compute or load param normalization
        # ------------------
        stats_path = os.path.join(dataset_path, "param_stats.npz")

        if split == 'train':
            corrected_list = []
            for img_path, raw_param in zip(self.image_paths, self.params):
                img_bytes = tf.io.read_file(img_path)
                img = tf.image.decode_png(img_bytes, channels=1)
                h = int(img.shape[0])
                w = int(img.shape[1])

                target_h, target_w = self.image_size
                scale_h = target_h / h
                scale_w = target_w / w
                resize_scale = min(scale_h, scale_w)

                corrected_list.append(raw_param[0] / resize_scale)

            corrected_arr = np.array(corrected_list).reshape(-1, 1)
            self.param_mean = corrected_arr.mean(0)
            self.param_std = corrected_arr.std(0)
            np.savez(stats_path, mean=self.param_mean, std=self.param_std)
        else:
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

    # -------------------------------------------------------
    # Parse function used by tf.data
    # -------------------------------------------------------
    def _parse_function(self, path, param, y):

        img_bytes = tf.io.read_file(path)
        image = tf.image.decode_png(img_bytes, channels=1)

        original_h = tf.cast(tf.shape(image)[0], tf.float32)
        original_w = tf.cast(tf.shape(image)[1], tf.float32)

        target_h, target_w = self.image_size
        scale_h = target_h / original_h
        scale_w = target_w / original_w
        resize_scale = tf.minimum(scale_h, scale_w)

        # Resize with padding
        image = tf.image.resize_with_pad(image, target_h, target_w)

        # Augmentation only on training
        if self.split == 'train':
            image = self._augment(image)

        image = tf.cast(image, tf.float32) / 255.0

        # Fix scale factor after resize
        corrected_param = param[0] / resize_scale

        # Normalize
        corrected_param = (corrected_param - self.param_mean) / self.param_std
        corrected_param = tf.reshape(corrected_param, (1,))

        return (image, corrected_param), y, path

    def _augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = self.translation_layer(tf.expand_dims(image, 0))[0]
        return image

    # -------------------------------------------------------
    # Build tf.data.Dataset
    # -------------------------------------------------------
    def get_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.params, self.outputs)
        )

        if self.shuffle and self.split == "train":
            ds = ds.shuffle(buffer_size=len(self.image_paths), seed=self.random_state)

        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
