import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import time

# Mixed precision for speed
tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: 
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
mixed_precision.set_global_policy("mixed_float16")

# Path for your generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from DataGeneratorv4 import ADSADataPipeline

# --------------------- Model ---------------------
def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=100):
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    # If grayscale, convert to RGB
    if input_image_shape[2] == 1:
        x_input = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(img_input)
    else:
        x_input = img_input

    base_model = EfficientNetB0(input_shape=(input_image_shape[0], input_image_shape[1], 3),
                                include_top=False, weights='imagenet')
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    x = base_model(x_input, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    combined = Concatenate()([x, param_input])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear', dtype='float32')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model

# --------------------- Training Script ---------------------
def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet model on ADSA dataset")
    parser.add_argument("--dataset_path", type=str, default="/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    use_cache = args.cache
    image_size = (800, 800)
    output_training = "Surface Tension (mN/m)"

    # ----- Create pipelines -----
    train_pipeline = ADSADataPipeline(dataset_path, split='train',
                                      image_size=image_size,
                                      output_type=output_training,
                                      batch_size=batch_size,
                                      shuffle=True)

    val_pipeline = ADSADataPipeline(dataset_path, split='val',
                                    image_size=image_size,
                                    output_type=output_training,
                                    batch_size=batch_size,
                                    shuffle=False)

    test_pipeline = ADSADataPipeline(dataset_path, split='test',
                                     image_size=image_size,
                                     output_type=output_training,
                                     batch_size=batch_size,
                                     shuffle=False)

    # Ensure val/test normalization matches train
    val_pipeline.param_mean = train_pipeline.param_mean
    val_pipeline.param_std  = train_pipeline.param_std
    test_pipeline.param_mean = train_pipeline.param_mean
    test_pipeline.param_std  = train_pipeline.param_std

    # ---------------- Prepare datasets ----------------
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare_for_training(ds):
        ds = ds.map(lambda inputs, y, _: (inputs, y), num_parallel_calls=AUTOTUNE)  # Drop filename
        if use_cache:
            ds = ds.cache()
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def prepare_for_eval(ds):
        ds = ds.map(lambda inputs, y, _: (inputs, y), num_parallel_calls=AUTOTUNE)
        if use_cache:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    # Get raw datasets
    raw_train_ds = train_pipeline.get_dataset()
    raw_val_ds   = val_pipeline.get_dataset()
    raw_test_ds  = test_pipeline.get_dataset()

    train_ds = prepare_for_training(raw_train_ds)
    val_ds = prepare_for_eval(raw_val_ds)
    test_ds = prepare_for_eval(raw_test_ds)

    # Save normalization info
    stats = {
        "param_mean": train_pipeline.param_mean.tolist(),
        "param_std": train_pipeline.param_std.tolist(),
        "image_size": list(image_size)
    }
    with open("SurfaceTension_Model_Large_Cust_V14_stats.json", "w") as f:
        json.dump(stats, f)

    # ----- Create & train model -----
    model = create_model(input_image_shape=(image_size[0], image_size[1], 3), input_param_size=2)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_SurfaceTensionENFv14.keras",
                                                       monitor="val_loss",
                                                       save_best_only=True)
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=50,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                            checkpoint_cb
                        ])

    model.save("SurfaceTensionENFv14.keras")

    # ----- Inference with filenames preserved -----
    all_true, all_pred, all_names, all_times = [], [], [], []
    print("[INFO] Running inference...")
    start_total = time.time()

    for ((X_batch, params_batch), y_batch, names_batch) in raw_test_ds:
        start = time.time()
        preds_batch = model.predict([X_batch, params_batch], verbose=0)
        elapsed = time.time() - start

        all_true.extend(y_batch.numpy().tolist())
        all_pred.extend(preds_batch.flatten().tolist())
        all_times.extend([elapsed / X_batch.shape[0]] * X_batch.shape[0])
        all_names.extend([n.numpy().decode('utf-8') for n in names_batch.numpy().reshape(-1)])

    results_df = pd.DataFrame({
        "image_name": all_names,
        "True_Value": all_true,
        "Predicted_Value": all_pred,
        "Prediction_Time_s": all_times
    })
    results_df.to_csv("ST_Model_Predictions.csv", index=False)
    print("[INFO] Results saved to ST_Model_Predictions.csv")

if __name__ == "__main__":
    main()
