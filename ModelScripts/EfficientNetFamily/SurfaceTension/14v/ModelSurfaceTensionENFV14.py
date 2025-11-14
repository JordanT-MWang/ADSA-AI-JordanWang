import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
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

from DataGeneratorv4 import ADSADataPipeline # your custom generator

def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=100):
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
    
    base_model = EfficientNetB0(input_shape=(input_image_shape[0], input_image_shape[1], 3), include_top=False, weights='imagenet')
   
    print("Base model input shape:", base_model.input_shape)
    # Freeze first N layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    x = base_model(img_input, training=True)
    x = GlobalAveragePooling2D()(x)

    # Custom trainable layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    # Concatenate with numeric input
    combined = Concatenate()([x, param_input])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear',dtype='float32')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model

def main():
 # parse to get local data set
    parser = argparse.ArgumentParser(description="Train EfficientNet model on ADSA dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined",
        help="Path to dataset directory containing Edges/, input_params.csv, output_params.csv"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache dataset in memory after map() (useful if dataset fits in RAM)"
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    use_cache = args.cache

    output_csv = "ST_Model_Predictions.csv"
    output_training = "Surface Tension (mN/m)"
    model_name = "SurfaceTensionENF4"
    image_size = (800, 800)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "best_SurfaceTensinoENFv12.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )

    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")

    # ----- Create pipelines -----
    train_pipeline = ADSADataPipeline(dataset_path,
                                      split='train',
                                      image_size=image_size,
                                      output_type=output_training,
                                      batch_size=batch_size,
                                      shuffle=True)

    # create val/test pipelines but DO NOT call get_dataset() yet
    val_pipeline = ADSADataPipeline(dataset_path,
                                    split='val',
                                    image_size=image_size,
                                    output_type=output_training,
                                    batch_size=batch_size,
                                    shuffle=False)

    test_pipeline = ADSADataPipeline(dataset_path,
                                     split='test',
                                     image_size=image_size,
                                     output_type=output_training,
                                     batch_size=batch_size,
                                     shuffle=False)

    # ----- Ensure normalization for val/test uses train stats -----
    # train_pipeline.param_mean/std are numpy arrays
    val_pipeline.param_mean = train_pipeline.param_mean
    val_pipeline.param_std  = train_pipeline.param_std
    test_pipeline.param_mean = train_pipeline.param_mean
    test_pipeline.param_std  = train_pipeline.param_std

    # ----- Build tf.data.Datasets -----
    # Note: our ADSADataPipeline.get_dataset() returns tuples ((image, params), label, filename)
    # Model.fit expects ( (img, params), label ). We'll strip filename and apply a few optimizations.

    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_for_training(ds, cache=use_cache, shuffle_buffer=2000):
        # ds: dataset that yields ((img, params), label, filename)
        ds = ds.map(lambda inputs, y, fn: (inputs, y),
                    num_parallel_calls=AUTOTUNE)  # drop filename
        if cache:
            # Cache decoded/processed tensors in RAM (fast) - use only if dataset fits in memory
            ds = ds.cache()
        # shuffle: small buffer is enough for good randomness, avoid huge memory use
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def prepare_for_eval(ds, cache=use_cache):
        ds = ds.map(lambda inputs, y, fn: (inputs, y), num_parallel_calls=AUTOTUNE)
        if cache:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    # Get raw datasets from your pipeline (they already batch inside get_dataset())
    raw_train_ds = train_pipeline.get_dataset()  # yields ((img, params), label, filename) batched
    raw_val_ds   = val_pipeline.get_dataset()
    raw_test_ds  = test_pipeline.get_dataset()

    # IMPORTANT: the pipelines returned batched datasets. Our prepare functions expect datasets that yield
    # ((img, params), y, filename) but still batched; map works element-wise on batches too.
    train_ds = prepare_for_training(raw_train_ds, cache=use_cache, shuffle_buffer=2000)
    val_ds = prepare_for_eval(raw_val_ds, cache=use_cache)
    test_ds = prepare_for_eval(raw_test_ds, cache=use_cache)

    # ----- Save normalization & image_size for reproducibility -----
    stats = {
        "param_mean": train_pipeline.param_mean.tolist(),
        "param_std": train_pipeline.param_std.tolist(),
        "image_size": list(image_size)
    }
    with open("SurfaceTension_Model_Large_Cust_V14_stats.json", "w") as f:
        json.dump(stats, f)

    # ----- Create model -----
    model = create_model(input_image_shape=(image_size[0], image_size[1], 3), input_param_size=2)

    # ----- Train -----
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=50,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                            checkpoint_cb
                        ])

    model.save("SurfaceTensionENFv14.keras")

    # ----- Evaluate & inference on test set -----
    test_loss, test_mae = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Inference loop that preserves image names and timing
    all_true = []
    all_pred = []
    all_times = []
    all_names = []

    print("[INFO] Running inference...")
    start_total = time.time()

    # raw_test_ds yields batches of ((X_batch, params_batch), y_batch, names_batch)
    for ((X_batch, params_batch), y_batch, names_batch) in raw_test_ds:
        start = time.time()
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed = time.time() - start

        all_true.extend(y_batch.numpy().tolist())
        all_pred.extend(preds_batch.flatten().tolist())
        all_times.extend([elapsed / X_batch.shape[0]] * X_batch.shape[0])
        # names_batch is a tf.Tensor of strings (batched)
        all_names.extend([n.numpy().decode('utf-8') for n in names_batch.numpy().reshape(-1)])

    end_total = time.time()
    total_elapsed_time = end_total - start_total
    print(f"Total inference time: {total_elapsed_time:.2f}s")

    results_df = pd.DataFrame({
        "image_name": all_names,
        "True_Value": all_true,
        "Predicted_Value": all_pred,
        "Prediction_Time_s": all_times
    })
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to {output_csv}")

    # Save plots (same as before)...
    plt.figure(figsize=(6,6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    if len(all_true) > 1:
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
    plt.xlabel("True Surface Tension")
    plt.ylabel("Predicted Tension")
    plt.title("Predicted vs Tension")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pred_vs_true_ST.png")
    plt.show()



if __name__ == "__main__":
    main()