import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/synthetic_sensor_data.csv")

X = df[["pulse_rate", "spo2", "temperature"]].values
y = df["stress_label"].values

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Normalize features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Reshape for CNN
# (samples, timesteps, channels)
# -----------------------------
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv1D(32, kernel_size=2, activation="relu", input_shape=(3, 1)),
    MaxPooling1D(pool_size=1),

    Conv1D(64, kernel_size=2, activation="relu"),
    MaxPooling1D(pool_size=1),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# -----------------------------
# Save model
# -----------------------------
model.save("cnn_model/physio_stress_cnn.h5")
print("💾 CNN model saved successfully")
