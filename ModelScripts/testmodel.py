import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

from DataGenerator import ADSADataGenerator # your custom generator
def run_inference(model_path, dataset_path, output_csv="predictions.csv", batch_size=16):
    # Load trained model
    # Explicitly provide custom_objects for standard loss/metrics if needed
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()})
    print(f"[INFO] Loaded model from {model_path}")

    # Initialize test generator
    test_gen = ADSADataGenerator(
        dataset_path,
        split='test',
        batch_size=batch_size,
        image_size=(512, 640),
        output_type='Surface Tension (mN/m)'
    )

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
        plt.savefig("pred_vs_true.png")
        plt.show()
    else:
        print("[INFO] No predictions were made, skipping plot generation.")


    return results_df if total_samples > 0 else pd.DataFrame(), avg_time

if __name__ == "__main__":
    model_path = "SurfaceTension_Model.h5"
    #dataset_path = "/content/drive/MyDrive/DataSetCombined"
    dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
    # Ensure ADSADataGenerator is defined by running the cell above this one first.
    run_inference(model_path, dataset_path)