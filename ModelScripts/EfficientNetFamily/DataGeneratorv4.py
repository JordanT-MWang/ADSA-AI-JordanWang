import tensorflow as tf
import numpy as np
import os
import pandas as pd

class ADSADataPipeline:
    def __init__(self, dataset_path, split='train', image_size=(512, 640),
                 output_type='Surface Tension', batch_size=32, shuffle=True, return_filenames=False, random_state=42):
        """
        return_filenames: whether to include filenames in the dataset (useful for inference)
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
        self.random_state = random_state
        self.return_filenames = return_filenames

        input_df = pd.read_csv(self.input_csv)
        output_df = pd.read_csv(self.output_csv)
        merged = pd.merge(input_df, output_df, on="Image Name", suffixes=("_in", "_out"))

        image_paths, params, outputs, filenames = [], [], [], []

        for _, row in merged.iterrows():
            path = os.path.join(self.image_dir, row["Image Name"])
            if os.path.exists(path):
                image_paths.append(path)
                params.append([row["Delta Rho (g/ml)"], row["Scale Factor (cm/pixel)"]])
                outputs.append(row[self.output_type])
                filenames.append(row["Image Name"])

        # Split into train/val/test
        from sklearn.model_selection import train_test_split
        train_idx, temp_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state=random_state)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

        if split == 'train': idx = train_idx
        elif split == 'val': idx = val_idx
        else: idx = test_idx

        self.image_paths = np.array(image_paths)[idx]
        self.params = np.array(params, dtype=np.float32)[idx]
        self.outputs = np.array(outputs, dtype=np.float32)[idx]
        self.filenames = np.array(filenames)[idx]

        # Compute normalization on train split
        if split == 'train':
            self.param_mean = np.mean(self.params, axis=0)
            self.param_std = np.std(self.params, axis=0)
        else:
            stats_path = os.path.join(dataset_path, "param_stats.npz")
            stats = np.load(stats_path)
            self.param_mean = stats["mean"]
            self.param_std = stats["std"]

        if split == 'train':
            np.savez(os.path.join(dataset_path, "param_stats.npz"),
                     mean=self.param_mean, std=self.param_std)

    def _parse_function(self, path, param, y, filename=None):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])

        # Normalize numeric params
        param = (param - self.param_mean) / self.param_std

        if self.split == 'train':
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.05)
            image = tf.image.random_contrast(image, 0.9, 1.1)

        if self.return_filenames:
            return (image, param), y, filename
        else:
            return (image, param), y

    def get_dataset(self):
        paths = tf.constant(self.image_paths, dtype=tf.string)
        params = tf.constant(self.params, dtype=tf.float32)
        labels = tf.constant(self.outputs, dtype=tf.float32)

        if self.return_filenames:
            filenames = tf.constant(self.filenames, dtype=tf.string)
            ds = tf.data.Dataset.from_tensor_slices((paths, params, labels, filenames))
            ds = ds.map(lambda p, pr, y, fn: self._parse_function(p, pr, y, fn),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices((paths, params, labels))
            ds = ds.map(lambda p, pr, y: self._parse_function(p, pr, y),
                        num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle and self.split == 'train':
            ds = ds.shuffle(buffer_size=len(self.image_paths))

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
