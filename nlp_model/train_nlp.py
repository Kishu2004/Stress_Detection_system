import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Expanded stress text dataset
# -----------------------------
data = {
    "text": [
        # Stress
        "I feel anxious and overwhelmed",
        "I am stressed about my exams",
        "My heart is racing and I feel nervous",
        "I cannot relax and feel pressure",
        "I feel very tense today",
        "I am worried about my future",
        "I feel exhausted and mentally drained",
        "There is too much pressure at work",
        "I feel panic and discomfort",
        "I am unable to focus due to stress",

        # No stress
        "I am calm and relaxed today",
        "I feel happy and peaceful",
        "Everything is going well",
        "I am feeling very comfortable",
        "I feel positive and confident",
        "Today is a good and relaxed day",
        "I am enjoying my time",
        "I feel content and satisfied",
        "I am mentally relaxed",
        "I feel balanced and calm"
    ],
    "label": [
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# Train-test split (STRATIFIED)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\n📊 NLP Classification Report:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"✅ NLP Model Accuracy: {acc * 100:.2f}%")

# -----------------------------
# Save
# -----------------------------
with open("nlp_model/text_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("nlp_model/text_stress_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 NLP model and vectorizer saved successfully")
