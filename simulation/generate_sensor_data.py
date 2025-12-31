import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

# Number of samples per class
NORMAL_SAMPLES = 1000
STRESS_SAMPLES = 1000

# -----------------------------
# Generate NORMAL data
# -----------------------------
normal_pulse = np.random.normal(loc=70, scale=5, size=NORMAL_SAMPLES)
normal_spo2 = np.random.normal(loc=98.5, scale=0.7, size=NORMAL_SAMPLES)
normal_temp = np.random.normal(loc=36.6, scale=0.2, size=NORMAL_SAMPLES)

normal_labels = np.zeros(NORMAL_SAMPLES)

# -----------------------------
# Generate STRESS data
# -----------------------------
stress_pulse = np.random.normal(loc=105, scale=8, size=STRESS_SAMPLES)
stress_spo2 = np.random.normal(loc=95.5, scale=0.8, size=STRESS_SAMPLES)
stress_temp = np.random.normal(loc=37.8, scale=0.3, size=STRESS_SAMPLES)

stress_labels = np.ones(STRESS_SAMPLES)

# -----------------------------
# Combine data
# -----------------------------
pulse = np.concatenate([normal_pulse, stress_pulse])
spo2 = np.concatenate([normal_spo2, stress_spo2])
temp = np.concatenate([normal_temp, stress_temp])
labels = np.concatenate([normal_labels, stress_labels])

# Create DataFrame
df = pd.DataFrame({
    "pulse_rate": pulse,
    "spo2": spo2,
    "temperature": temp,
    "stress_label": labels
})

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv("data/synthetic_sensor_data.csv", index=False)

print("✅ Synthetic sensor data generated successfully!")
print(df.head())

# -----------------------------
# Visualization (for report)
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df[df["stress_label"] == 0]["pulse_rate"], bins=30, alpha=0.6, label="Normal")
plt.hist(df[df["stress_label"] == 1]["pulse_rate"], bins=30, alpha=0.6, label="Stress")
plt.title("Pulse Rate Distribution")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(df[df["stress_label"] == 0]["spo2"], bins=30, alpha=0.6, label="Normal")
plt.hist(df[df["stress_label"] == 1]["spo2"], bins=30, alpha=0.6, label="Stress")
plt.title("SpO₂ Distribution")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(df[df["stress_label"] == 0]["temperature"], bins=30, alpha=0.6, label="Normal")
plt.hist(df[df["stress_label"] == 1]["temperature"], bins=30, alpha=0.6, label="Stress")
plt.title("Temperature Distribution")
plt.legend()

plt.tight_layout()
plt.show()
