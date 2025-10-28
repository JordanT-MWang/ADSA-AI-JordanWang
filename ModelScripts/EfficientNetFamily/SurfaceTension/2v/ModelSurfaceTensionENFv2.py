import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import mixed_precision
tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: 
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
#mixed_precision.set_global_policy("mixed_float16")

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import os # Import os module
import sys
import json
# === Path handling for DataGenerator ===
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from DataGeneratorv2 import ADSADataGenerator # your custom generator

def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=75):
    """
    MobileNetV2 for regression with numeric inputs.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")
    print(input_image_shape)
    # If somehow input channels=1, repeat to 3
    if input_image_shape[2] == 1:
        x_input = Concatenate()([img_input, img_input, img_input])
    else:
        x_input = img_input
    # Load pretrained MobileNetV2
    base_model = EfficientNetB2(input_shape=(input_image_shape[0], input_image_shape[1], 3), include_top=False, weights='imagenet')
   
    print("Base model input shape:", base_model.input_shape)
    # Freeze first N layers
    #for i, layer in enumerate(base_model.layers):
    #    layer.trainable = i >= freeze_until

    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Custom trainable layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    # Concatenate with numeric input
    combined = Concatenate()([x, param_input])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
    return model

def main():
    #dataset_path = "/content/drive/MyDrive/DataSetCombined"
    dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
    output_csv = "ST_Model_Predictions.csv"
    batch_size = 64
    image_size = (512, 512)

    # Print paths for debugging
    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")

    
    train_gen = ADSADataGenerator(dataset_path, split='train', batch_size=batch_size,
                              image_size=image_size, output_type='Surface Tension (mN/m)')
    
    val_gen = ADSADataGenerator(dataset_path, split='val', batch_size=batch_size,
                                image_size=image_size, output_type='Surface Tension (mN/m)')
    test_gen = ADSADataGenerator(dataset_path, split='test', batch_size=batch_size,
                                image_size=image_size, output_type='Surface Tension (mN/m)')
    #test input shape
    (X_img, X_params), y = train_gen[0]
    print("Image batch shape:", X_img.shape)
    print("Parameter batch shape:", X_params.shape)
    print("Output batch shape:", y.shape)
 
    # Model now expects 1 for channel for custom and 3 for mobilenet
    model = create_model(input_image_shape=(512, 512, 3), input_param_size=2)
    print(model.input_shape)
    base_model = model.get_layer('efficientnetb2')
    print(base_model.layers[0].name, base_model.layers[0].input_shape, base_model.layers[0].output_shape)
    # Save normalization statistics for future inference
    if ADSADataGenerator.param_mean is not None:
        """
        model._metadata = {
        "param_mean": ADSADataGenerator.param_mean.tolist() if ADSADataGenerator.param_mean is not None else None,
        "param_std": ADSADataGenerator.param_std.tolist() if ADSADataGenerator.param_std is not None else None,
    }"""
        stats = {
        "param_mean": ADSADataGenerator.param_mean.tolist(),
        "param_std": ADSADataGenerator.param_std.tolist(),
        }
        with open("SurfaceTension_Model_Large_Cust_V1_stats.json", "w") as f:
            json.dump(stats, f)
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    # Save model
    model.save("SurfaceTensionENFv1.keras")

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
    plt.savefig("training_curves_ST.png")
    plt.show()

    all_true = []
    all_pred = []
    all_times = []
    all_names = []  # <-- Track image names

    print("[INFO] Running inference...")

    start_total = time.time() # Start timing for the whole inference process

    # Go through test batches and predict on the whole batch
    for batch_idx, ((X_batch, params_batch), y_batch) in enumerate(test_gen):
        start_batch = time.time()
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed_batch = time.time() - start_batch

        all_true.extend(y_batch)
        all_pred.extend(preds_batch.flatten())
        all_times.extend([elapsed_batch / len(y_batch)] * len(y_batch))

        # Correct way to get image names
        start_idx = batch_idx * test_gen.batch_size
        end_idx = start_idx + len(y_batch)
        all_names.extend(test_gen.image_list[start_idx:end_idx])


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
            "image_name": all_names,
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
        plt.savefig("pred_vs_true_ST.png")
        plt.show()
    else:
        print("[INFO] No predictions were made, skipping plot generation.")


if __name__ == "__main__":
    main()