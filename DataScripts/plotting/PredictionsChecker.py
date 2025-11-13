#model keeps having the same data spreed. using this script to test and see
#images that keep on being scattered to dtm issue
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
input_csv = r"E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\EfficientNetFamily\SurfaceTension\10v\ST_Model_Predictions.csv"
output_csv = r"E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\EfficientNetFamily\SurfaceTension\10v\ST_Model_Predictions_Analyzed_etf_v10.csv"
output_plot = r"E:\Data\Jordan\Jordan Wang AI-ADSA\ADSA-AI-JordanWang\ModelScripts\EfficientNetFamily\SurfaceTension\10v\pred_vs_true_with_zones.png"

# === LOAD DATA ===
df = pd.read_csv(input_csv)

# Convert Tensor strings like "tf.Tensor(22.09, shape=(), dtype=float32)" → float
def extract_float(value):
    if isinstance(value, str) and "tf.Tensor" in value:
        try:
            return float(value.split("(")[1].split(",")[0])
        except:
            return np.nan
    return float(value)

df["True_Value"] = df["True_Value"].apply(extract_float)
df["Predicted_Value"] = df["Predicted_Value"].apply(extract_float)
# === DEFINE PARALLEL ±% OFFSETS ===
mean_true = np.mean(df["True_Value"])
offset_10 = 0.10 * mean_true
offset_20 = 0.20 * mean_true
# === COMPUTE PERCENTAGE ERROR ===
df["Percent_Error"] = abs(df["Predicted_Value"] - df["True_Value"]) 
# === CLASSIFY ZONES ===
def classify_zone(err):
    if err <= offset_10:
        return "black"   # within ±5%
    elif err <= offset_20:
        return "orange"  # between 5–10%
    else:
        return "red"     # outside ±10%

df["Zone"] = df["Percent_Error"].apply(classify_zone)

# === SAVE UPDATED CSV ===
df.to_csv(output_csv, index=False)
print(f"[INFO] Saved analyzed results to: {output_csv}")

# === PLOT ===
plt.figure(figsize=(7,7))
plt.title("Predicted vs True Values with ±10% and ±20% Zones Surface Tension")

x = np.linspace(min(df["True_Value"]), max(df["True_Value"]), 200)

plt.plot(x, x, 'k--', label="Ideal")

# Use offsets (parallel lines)
value_range = max(df["True_Value"]) - min(df["True_Value"])


plt.plot(x, x + offset_10, 'g--', label="+10%")
plt.plot(x, x - offset_10, 'g--')
plt.plot(x, x + offset_20, 'r--', label="+20%")
plt.plot(x, x - offset_20, 'r--')

# Scatter points by zone
for color in ["black", "orange", "red"]:
    subset = df[df["Zone"] == color]
    plt.scatter(subset["True_Value"], subset["Predicted_Value"], color=color, alpha=0.6, label=f"{color} zone")

plt.xlabel("True Value Surface Tension [mN/m]")
plt.ylabel("Predicted Value Surface Tension [mN/m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"[INFO] Plot saved as: {output_plot}")
