import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ADSADataGenerator import ADSADataGenerator  # your custom generator

def run_inference(model_path, dataset_path, output_csv="predictions.csv", batch_size=16):
    # Load trained model
    model = load_model(model_path)
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

    # Go through test batches
    for (X_batch, params_batch), y_batch in test_gen:
        for i in range(len(y_batch)):
            x_img = np.expand_dims(X_batch[i], axis=0)
            x_param = np.expand_dims(params_batch[i], axis=0)

            start = time.time()
            pred = model.predict([x_img, x_param], verbose=0)
            elapsed = time.time() - start

            all_true.append(y_batch[i])
            all_pred.append(pred.flatten()[0])
            all_times.append(elapsed)

    # Summary statistics
    avg_time = np.mean(all_times)
    print(f"[INFO] Inference complete.")
    print(f"    Total samples: {len(all_true)}")
    print(f"    Avg prediction time: {avg_time*1000:.3f} ms")
    print(f"    Min/Max time: {min(all_times)*1000:.3f} / {max(all_times)*1000:.3f} ms")

    # Save results to CSV
    results_df = pd.DataFrame({
        "True_Value": all_true,
        "Predicted_Value": all_pred,
        "Prediction_Time_s": all_times
    })
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to {output_csv}")

    # Plot predicted vs true values
    plt.figure(figsize=(6,6))
    plt.scatter(all_true, all_pred, alpha=0.6)
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
    plt.xlabel("True Surface Tension (mN/m)")
    plt.ylabel("Predicted Surface Tension (mN/m)")
    plt.title("Predicted vs True Values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pred_vs_true.png")
    plt.show()

    return results_df, avg_time

if __name__ == "__main__":
    model_path = "SurfaceTension_Model.h5"
    dataset_path = "/content/drive/MyDrive/DataSetCombined"
    run_inference(model_path, dataset_path)
