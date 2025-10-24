import os
import sys

# === Must set BEFORE importing tensorflow ===
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# (optional) Disable mixed precision if it's unstable
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"

# === Now import tensorflow and keras ===
import tensorflow as tf
from tensorflow.keras import mixed_precision
tf.config.optimizer.set_jit(False)

# Set GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# You can re-enable mixed precision if your model benefits from it


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate,
    Conv2D, BatchNormalization, MaxPooling2D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd

# === Path handling for DataGenerator ===
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from CustomCNNDataGenerator import CustomCNNADSADataGenerator # your custom generator

def create_custom_cnn(input_image_shape=(512, 640, 1), input_param_size=2):
    """
    A slightly smaller CNN for regression with numeric inputs.
    Lighter and faster than the original, still expressive.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    # --- Conv Block 1 ---
    x = Conv2D(16, 3, activation='relu', padding='same')(img_input)  # reduced filters
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.05)(x)

    # --- Conv Block 2 ---
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.1)(x)

    # --- Conv Block 3 ---
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.15)(x)

    # --- Conv Block 4 ---
    x = Conv2D(64, 3, activation='relu', padding='same')(x)  # reduced last block
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # --- Dense head ---
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)

    # --- Combine with numeric parameters ---
    combined = Concatenate()([x, param_input])
    z = Dense(16, activation='relu')(combined)
    z = Dropout(0.05)(z)
    z = Dense(8, activation='relu')(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
    return model

def main():
    #dataset_path = "/content/drive/MyDrive/DataSetCombined"
    dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
    output_csv = "ST_Model_Predictions_Cust.csv"
    batch_size = 32
    image_size = (512, 640)

    # Print paths for debugging
    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")

    
    train_gen = CustomCNNADSADataGenerator(dataset_path, split='train', batch_size=batch_size,
                                    image_size=image_size, output_type='Surface Tension (mN/m)')

    val_gen = CustomCNNADSADataGenerator(dataset_path, split='val', batch_size=batch_size,
                                  image_size=image_size, output_type='Surface Tension (mN/m)')

    test_gen = CustomCNNADSADataGenerator(dataset_path, split='test', batch_size=batch_size,
                                   image_size=image_size, output_type='Surface Tension (mN/m)')
                                  

    # Quick generator sanity check
    (X_batch, params_batch), y_batch = train_gen[0]  # get first batch from generator __getitem__
    print("X batch shape, dtype:", getattr(X_batch, "shape", None), getattr(X_batch, "dtype", None))
    print("params batch shape, dtype:", getattr(params_batch, "shape", None), getattr(params_batch, "dtype", None))
    print("y batch shape, dtype:", getattr(y_batch, "shape", None), getattr(y_batch, "dtype", None))
    # Ensure channel dim exists
    assert X_batch.ndim == 4 and X_batch.shape[-1] in (1,3), "Image batch must be (B,H,W,C) with C=1 or 3"
    assert X_batch.dtype == np.float32 or X_batch.dtype == np.uint8, "Prefer float32 or uint8"
    # Model now expects 1 for channel for custom and 3 for mobilenet
    model = create_custom_cnn(input_image_shape=(512, 640, 1), input_param_size=2)
    # Save normalization statistics for future inference
    if CustomCNNADSADataGenerator.param_mean is not None:
        model._metadata = {
        "param_mean": CustomCNNADSADataGenerator.param_mean.tolist() if CustomCNNADSADataGenerator.param_mean is not None else None,
        "param_std": CustomCNNADSADataGenerator.param_std.tolist() if CustomCNNADSADataGenerator.param_std is not None else None,
    }
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Save model
    model.save("SurfaceTension_Model_Large_Cust_V1.keras")

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Plot training/validation curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves_ST_Cust.png")
    plt.show()

    all_true = []
    all_pred = []
    all_times = []

    print("[INFO] Running inference...")

    start_total = time.time() # Start timing for the whole inference process

    # Go through test batches and predict on the whole batch
    for (X_batch, params_batch), y_batch in test_gen:
        start_batch = time.time() # Start timing for the batch prediction
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed_batch = time.time() - start_batch # Time for the batch prediction

        # Extend the lists with batch results
        all_true.extend(y_batch)
        all_pred.extend(preds_batch.flatten())
        # For simplicity, we'll record the batch time for each sample in the batch
        # A more precise timing would require predicting samples individually, which is slow
        all_times.extend([elapsed_batch / len(y_batch)] * len(y_batch)) # Avg time per sample in this batch


    end_total = time.time() # End timing for the whole inference process
    total_elapsed_time = end_total - start_total

    # Summary statistics
    avg_time = np.mean(all_times) if all_times else 0 # Handle case with no predictions
    total_samples = len(all_true)
    print(f"[INFO] Inference complete.")
    print(f"    Total samples predicted: {total_samples}")
    print(f"    Total inference time: {total_elapsed_time:.3f} seconds")
    if total_samples > 0:
        print(f"    Avg prediction time per sample: {avg_time*1000:.3f} ms")


    # Save results to CSV
    if total_samples > 0:
        results_df = pd.DataFrame({
            "True_Value": all_true,
            "Predicted_Value": all_pred,
            "Prediction_Time_s": all_times
        })
        results_df.to_csv(output_csv, index=False)
        print(f"[INFO] Results saved to {output_csv}")
    else:
        print("[INFO] No predictions were made, skipping CSV save.")


    # Plot predicted vs true values
    if total_samples > 0:
        plt.figure(figsize=(6,6))
        plt.scatter(all_true, all_pred, alpha=0.6)
        # Ensure there are enough points for the line
        if len(all_true) > 1:
            plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
        plt.xlabel("True Surface Tension (mN/m)")
        plt.ylabel("Predicted Surface Tension (mN/m)")
        plt.title("Predicted vs True Values")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pred_vs_true_ST_Cust.png")
        plt.show()
    else:
        print("[INFO] No predictions were made, skipping plot generation.")


if __name__ == "__main__":
    main()