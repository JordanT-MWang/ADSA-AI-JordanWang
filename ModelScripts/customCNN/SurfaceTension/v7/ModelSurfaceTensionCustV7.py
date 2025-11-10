import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Lambda, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import mixed_precision
tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: 
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
mixed_precision.set_global_policy("mixed_float16")


import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import os # Import os module
import sys
import json
import argparse

# === Path handling for DataGenerator ===
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from DataGeneratorv3 import ADSADataPipeline # your custom generator

def conv_block(x, filters, kernel_size=3, pool=True, dropout=0.0, activation='relu', bn=True):
    """Reusable convolutional block."""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    if bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if pool:
        x = MaxPooling2D(pool_size=2)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x

def residual_block(x, filters, kernel_size=3, activation='relu'):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x
def create_custom_cnn(input_image_shape=(512, 640, 1), input_param_size=2):
    """
    A slightly smaller CNN for regression with numeric inputs.
    Lighter and faster than the original, still expressive.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    x = conv_block(img_input, 16, dropout=0.1)
    x = conv_block(x, 32, dropout=0.15)
    x = conv_block(x, 64, dropout=0.25)
    x = conv_block(x, 128, dropout=0.3)
    x = GlobalAveragePooling2D()(x)
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
    #parse to get local data set
    parser = argparse.ArgumentParser(description="Train EfficientNet model on ADSA dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined",
        help="Path to dataset directory containing Edges/, input_params.csv, output_params.csv"
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path  # Use whatever was passed in
    
    output_csv = "ST_Model_Predictions.csv"
    output_training = "Surface Tension (mN/m)"
    batch_size = 64
    model_name="SurfaceTensionENF4"
    image_size = (510, 384)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_SurfaceTensinoENFv10.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    )
    # Print paths for debugging
    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")
                                  

    train_pipeline = ADSADataPipeline(dataset_path, split='train',image_size=image_size, output_type=output_training, batch_size=batch_size)
    val_gen = ADSADataPipeline(dataset_path, split='val',image_size=image_size, output_type=output_training, batch_size=batch_size).get_dataset()
    test_gen = ADSADataPipeline(dataset_path, split='test',image_size=image_size, output_type=output_training, batch_size=batch_size).get_dataset()
    train_gen = train_pipeline.get_dataset()

    # Save normalization stats
    stats = {
        "param_mean": train_pipeline.param_mean.tolist(),
        "param_std": train_pipeline.param_std.tolist(),
        "image_size": list(image_size)  # Save as list to be JSON serializable
    }
    with open("SurfaceTension_Model_Large_Cust_V10_stats.json", "w") as f:
        json.dump(stats, f)
    model = create_custom_cnn(input_image_shape=(image_size[0], image_size[1], 1), input_param_size=2)
    # Save normalization statistics for future inference
    
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        checkpoint_cb])

    # Save model
    model.save("SurfaceTension_Model_Large_Cust_V5.keras")

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