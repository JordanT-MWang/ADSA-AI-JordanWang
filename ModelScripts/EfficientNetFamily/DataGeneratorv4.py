import tensorflow as tf
import pandas as pd
import os
from glob import glob

class ADSADataPipeline:
    def __init__(self, dataset_path, split="train",
                 image_size=(800, 800),
                 batch_size=32,
                 output_type="Surface Tension (mN/m)",
                 shuffle=True):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_type = output_type

        # ---------------------------
        # Load CSVs
        # ---------------------------
        self.input_df = pd.read_csv(os.path.join(dataset_path, "input_params.csv"))
        self.output_df = pd.read_csv(os.path.join(dataset_path, "output_params.csv"))

        # Filter split
        self.input_df = self.input_df[self.input_df["split"] == split]
        self.output_df = self.output_df[self.output_df["split"] == split]

        # ---------------------------
        # SORT BOTH BY FILENAME = 0 MISALIGNMENT
        # ---------------------------
        self.input_df = self.input_df.sort_values("filename").reset_index(drop=True)
        self.output_df = self.output_df.sort_values("filename").reset_index(drop=True)

        # Make sure alignment is perfect
        assert list(self.input_df["filename"]) == list(self.output_df["filename"]), \
            "❌ INPUT & OUTPUT CSVs NOT ALIGNED — FILENAMES DO NOT MATCH!"

        # Keep paths as a list
        self.image_paths = [
            os.path.join(dataset_path, "Edges", fname)
            for fname in self.input_df["filename"]
        ]

        # Targets
        self.y = self.output_df[self.output_type].astype("float32").values

        # Two numeric params (same as old generator)
        self.params = self.input_df[["param1", "param2"]].astype("float32").values

        # Compute normalization (train only)
        if split == "train":
            self.param_mean = self.params.mean(axis=0)
            self.param_std = self.params.std(axis=0) + 1e-7
        else:
            # Will be loaded from training pipeline
            self.param_mean = None
            self.param_std = None

    # ------------------------------------------------------------------------------
    # TF ops
    # ------------------------------------------------------------------------------
    def _load_png(self, path):
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _tf_load(self, image_path, params, label, filename):
        image = self._load_png(image_path)

        # Normalize numeric params (done in pipeline)
        if self.param_mean is not None:
            params = (params - self.param_mean) / self.param_std

        return (image, params, filename), label

    # ------------------------------------------------------------------------------
    # Build tf.data.Dataset
    # ------------------------------------------------------------------------------
    def get_dataset(self):
        paths = tf.constant(self.image_paths, dtype=tf.string)
        params = tf.constant(self.params, dtype=tf.float32)
        labels = tf.constant(self.y, dtype=tf.float32)
        filenames = tf.constant(self.input_df["filename"].values, dtype=tf.string)

        ds = tf.data.Dataset.from_tensor_slices((paths, params, labels, filenames))

        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(self.image_paths), reshuffle_each_iteration=True)

        ds = ds.map(
            lambda p, pr, y, fn: self._tf_load(p, pr, y, fn),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
