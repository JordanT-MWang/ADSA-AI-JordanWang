import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

class ADSADataGenerator(Sequence):
    param_mean = None
    param_std = None

    def __init__(self, dataset_path, split='train', batch_size=32, image_size=(512, 640),
                 output_type='Surface Tension', shuffle=True, random_state=42, **kwargs):
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "Edges")
        self.input_csv = os.path.join(dataset_path, "input_params.csv")
        self.output_csv = os.path.join(dataset_path, "output_params.csv")

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.output_type = output_type
        self.random_state = random_state

        # Load CSVs
        self.input_df = pd.read_csv(self.input_csv)
        self.output_df = pd.read_csv(self.output_csv)

        # Build dictionaries for lookup by exact image name
        self.input_dict = {
            row['Image Name']: row[['Delta Rho (g/ml)', 'Scale Factor (cm/pixel)']].values.astype(np.float32)
            for _, row in self.input_df.iterrows()
        }

        self.output_dict = {
            row['Image Name']: row[self.output_type]
            for _, row in self.output_df.iterrows()
        }

        # Build master list of image names that exist in the folder
        self.image_list = []
        for img_name in self.output_dict.keys():
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):
                self.image_list.append(img_name)
            else:
                print(f"[WARNING] Missing image file: {img_path}")
        # Keep only the first half of the dataset
        #half_len = len(self.image_list) // 2
        #self.image_list = self.image_list[:half_len]

        # Split data (80% train, 10% val, 10% test)
        train_names, temp_names = train_test_split(
            self.image_list, test_size=0.2, random_state=self.random_state
        )
        val_names, test_names = train_test_split(
            temp_names, test_size=0.5, random_state=self.random_state
        )

        if split == 'train':
            self.image_list = train_names
        elif split == 'val':
            self.image_list = val_names
        elif split == 'test':
            self.image_list = test_names
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Compute normalization stats once (from training data only)
        if ADSADataGenerator.param_mean is None and split == 'train':
            params = np.array([self.input_dict[img] for img in train_names if img in self.input_dict])
            ADSADataGenerator.param_mean = params.mean(axis=0)
            ADSADataGenerator.param_std = params.std(axis=0)
            print(f"[INFO] Computed normalization stats: mean={ADSADataGenerator.param_mean}, std={ADSADataGenerator.param_std}")

        self.indexes = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_list) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = [self.image_list[i] for i in batch_indexes]

        X_img, X_input, y = [], [], []

        for img_file in batch_images:
            img_path = os.path.join(self.image_dir, img_file)
            image = self._load_and_pad(img_path)

            params = self.input_dict.get(img_file)
            output = self.output_dict.get(img_file)

            if params is not None and output is not None:
                # Normalize params
                params = (params - ADSADataGenerator.param_mean) / ADSADataGenerator.param_std
                X_img.append(image)
                X_input.append(params)
                y.append(output)

        return (np.array(X_img, dtype=np.float32), np.array(X_input, dtype=np.float32)), np.array(y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_and_pad(self, path):
        """Resize + pad image to target size without cropping and convert to 3 channels."""
        img = Image.open(path).convert("L")  # grayscale
        img = img_to_array(img)

        target_h, target_w = self.image_size
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=0)
        img = img / 255.0

        # Convert 1-channel grayscale -> 3-channel
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # shape: (H, W, 3)
        return img
