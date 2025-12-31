import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load CNN model
# -----------------------------
cnn_model = tf.keras.models.load_model(
    "cnn_model/physio_stress_cnn.h5"
)

# -----------------------------
# Load NLP model + vectorizer
# -----------------------------
with open("nlp_model/text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("nlp_model/text_stress_model.pkl", "rb") as f:
    nlp_model = pickle.load(f)

# -----------------------------
# Example INPUT (simulate sensor + text)
# -----------------------------
pulse_rate = 102       # BPM
spo2 = 95.2            # %
temperature = 37.9     # °C

user_text = "I feel anxious and under a lot of pressure"

# -----------------------------
# Prepare physiological input
# -----------------------------
X_physio = np.array([[pulse_rate, spo2, temperature]])

# Scale using new scaler (demo purpose)
scaler = StandardScaler()
X_physio = scaler.fit_transform(X_physio)

# Reshape for CNN
X_physio = X_physio.reshape(1, 3, 1)

# -----------------------------
# CNN Prediction
# -----------------------------
cnn_prob = cnn_model.predict(X_physio)[0][0]

# -----------------------------
# NLP Prediction
# -----------------------------
text_vec = vectorizer.transform([user_text])
nlp_prob = nlp_model.predict_proba(text_vec)[0][1]

# -----------------------------
# Fusion Logic
# -----------------------------
final_score = (0.7 * cnn_prob) + (0.3 * nlp_prob)

# -----------------------------
# Final Decision
# -----------------------------
if final_score >= 0.5:
    stress_state = "STRESSED"
else:
    stress_state = "NORMAL"

# -----------------------------
# Output
# -----------------------------
print("🧠 Physiological Stress Probability (CNN):", round(cnn_prob, 3))
print("📝 Text Stress Probability (NLP):", round(nlp_prob, 3))
print("🔀 Final Fused Score:", round(final_score, 3))
print("✅ FINAL STRESS STATE:", stress_state)
