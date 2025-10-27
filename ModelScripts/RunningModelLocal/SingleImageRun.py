import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------------
# Load model
# -------------------------
model_path = "E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\MobileNet\SurfaceTension\SurfaceTension_Model_Large_Mobile_V1.keras"
model = load_model(model_path, compile=False)

# Retrieve normalization stats if saved in model metadata
param_mean = np.array([9.9271584e-01, 6.0119742e-04])
param_std = np.array([0.00061441, 0.00056276])

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_image(img_path, target_size=(512, 640)):
    """Resize + pad like in the generator."""
    img = Image.open(img_path).convert("L")
    img = img_to_array(img)

    target_h, target_w = target_size
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

    # Convert to 3 channels
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img.astype(np.float32)

# -------------------------
# Inference function
# -------------------------
def predict(img_path, params):
    img = preprocess_image(img_path)
    params = np.array(params, dtype=np.float32)
    params = (params - param_mean) / param_std  # normalize
    params = np.expand_dims(params, axis=0)

    pred = model.predict([img, params])
    return pred[0][0]

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    image_path = r"E:\Data\Jordan\Jordan Wang AI-ADSA\Dataset\from lobb backup\re_organizedfromlobb\6-Amanda\1a_200_286\Edges\adsorption. 6.00_edge.png"  # path to your image
    params = [0.99333, 0.0006]               # [Delta Rho (g/ml), Scale Factor (cm/pixel)]

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
    else:
        prediction = predict(image_path, params)
        print(f"Predicted Surface Tension (mN/m): {prediction:.4f}")
