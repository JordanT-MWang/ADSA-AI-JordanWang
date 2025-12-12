import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Lambda
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

def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=25):
    """
    MobileNetV2 for regression with numeric inputs.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")
    
    # If somehow input channels=1, repeat to 3
    
    if input_image_shape[2] == 1:
        x_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(img_input)
    else:
        x_input = img_input
    # Load pretrained MobileNetV2
    
    base_model = EfficientNetB1(input_shape=(input_image_shape[0], input_image_shape[1], 3), include_top=False, weights='imagenet')
   
    print("Base model input shape:", base_model.input_shape)
    # Freeze first N layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Custom trainable layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    # Concatenate with numeric input
    combined = Concatenate()([x, param_input])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, activation='linear',dtype='float32')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
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
    batch_size = 128
    model_name="SurfaceTensionENF4"
    image_size = (640, 640)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_TensionENFv1.keras",
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
    with open("SurfaceTension_Model_Large_ENF_V1_stats.json", "w") as f:
        json.dump(stats, f)


    # Model now expects 1 for channel for custom and 3 for mobilenet
    model = create_model(input_image_shape=(image_size[0], image_size[1], 3), input_param_size=1)



    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        checkpoint_cb])

    # Save model
    model.save("TensionV1ENF.keras")

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
    plt.savefig("training_curves_Area.png")
    plt.show()

    all_true = []
    all_pred = []
    all_times = []
    all_names = []

    print("[INFO] Running inference...")

    start_total = time.time()

    # Keep track of index manually using the pipeline
    test_pipeline = ADSADataPipeline(dataset_path, split='test',image_size=image_size, output_type=output_training, batch_size=batch_size)
    test_gen = test_pipeline.get_dataset()
    image_paths = test_pipeline.image_paths  # Original list of image paths

    for batch_idx, ((X_batch, params_batch), y_batch) in enumerate(test_gen):
        start_batch = time.time()
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed_batch = time.time() - start_batch

        all_true.extend(y_batch.numpy().flatten())
        all_pred.extend(preds_batch.flatten())
        all_times.extend([elapsed_batch / len(y_batch)] * len(y_batch))

        # Use the batch index to get correct image names
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(y_batch)
        all_names.extend(image_paths[start_idx:end_idx])

    end_total = time.time()
    total_elapsed_time = end_total - start_total

    # Save results
    results_df = pd.DataFrame({
        "image_name": all_names,
        "True_Value": all_true,
        "Predicted_Value": all_pred,
        "Prediction_Time_s": all_times
    })
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to {output_csv}")

    # Optional: predicted vs true plot
    plt.figure(figsize=(6,6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    if len(all_true) > 1:
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
    plt.xlabel("True Surface Tension (mN/m)")
    plt.ylabel("Predicted Surface Tension (mN/m)")
    plt.title("Predicted vs True Values Surface Tension (mN/m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pred_vs_true_Tension.png")
    plt.show()



if __name__ == "__main__":
    main()