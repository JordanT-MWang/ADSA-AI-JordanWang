import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Flatten, EarlyStopping, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import pandas as pd
import os # Import os module

from DataGenerator import ADSADataGenerator # your custom generator
from CustomCNNDataGenerator import CustomCNNADSADataGenerator
def create_custom_cnn(input_image_shape=(512, 640, 1), input_param_size=2):
    """
    A deeper CNN for regression with numeric inputs.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    # --- Conv Block 1 ---
    x = Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)

    # --- Conv Block 2 ---
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # --- Conv Block 3 ---
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # --- Conv Block 4 ---
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.4)(x)

    # --- Flatten or Global Pooling ---
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)

    # --- Combine with numeric input ---
    combined = Concatenate()([x, param_input])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu')(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

    return model
def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=100):
    """
    MobileNetV2 for regression with numeric inputs.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(input_shape=input_image_shape, include_top=False, weights='imagenet')

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
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model

def main():
    #dataset_path = "/content/drive/MyDrive/DataSetCombined"
    dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
    output_csv = "Curvature_Model_Predictions.csv"
    batch_size = 16
    image_size = (512, 640)

    # Print paths for debugging
    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")

    """
    train_gen = ADSADataGenerator(dataset_path, split='train', batch_size=batch_size,
                              image_size=image_size, output_type='Curvature (1/cm)')
    
    val_gen = ADSADataGenerator(dataset_path, split='val', batch_size=batch_size,
                                image_size=image_size, output_type='Curvature (1/cm)')
    test_gen = ADSADataGenerator(dataset_path, split='test', batch_size=batch_size,
                                image_size=image_size, output_type='Curvature (1/cm)')
    """
    train_gen = CustomCNNADSADataGenerator(dataset_path, split='train', batch_size=batch_size,
                                    image_size=image_size, output_type='Curvature (1/cm)')

    val_gen = CustomCNNADSADataGenerator(dataset_path, split='val', batch_size=batch_size,
                                  image_size=image_size, output_type='Curvature (1/cm)')

    test_gen = CustomCNNADSADataGenerator(dataset_path, split='test', batch_size=batch_size,
                                   image_size=image_size, output_type='Curvature (1/cm)')
    # Model now expects 1 for channel for custom and 3 for mobilenet
    model = create_custom_cnn(input_image_shape=(512, 640, 1), input_param_size=2)
    # Save normalization statistics for future inference
    if ADSADataGenerator.param_mean is not None:
        model._metadata = {
        "param_mean": ADSADataGenerator.param_mean.tolist() if ADSADataGenerator.param_mean is not None else None,
        "param_std": ADSADataGenerator.param_std.tolist() if ADSADataGenerator.param_std is not None else None,
    }
    # Callbacks for better monitoring
    csv_logger = CSVLogger("training_log.csv", append=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Print progress nicely during training
    def on_epoch_end(epoch, logs):
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{time_str}] Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, mae={logs['mae']:.4f}, val_mae={logs['val_mae']:.4f}")

    progress_logger = LambdaCallback(on_epoch_end=on_epoch_end)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop, csv_logger, progress_logger],
        verbose=0  # use custom print instead
    )

    # Save model
    model.save("Curvature_Model.keras")

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
    plt.savefig("training_curves.png")
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
        plt.xlabel("True Curvature (1/cm)")
        plt.ylabel("Predicted Curvature (1/cm)")
        plt.title("Predicted vs True Values")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pred_vs_true.png")
        plt.show()
    else:
        print("[INFO] No predictions were made, skipping plot generation.")


if __name__ == "__main__":
    main()