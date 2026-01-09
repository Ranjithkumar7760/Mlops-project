"""
SMS Spam Detection - Training Script
MLOps-ready with:
- DVC data versioning
- MLflow experiment tracking
- Quality gate enforcement
"""

import pandas as pd
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================
# PATHS
# ============================
DATA_PATH = "data/spam.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# MLFLOW CONFIG
# ============================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "SMS-Spam-Detection"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"📡 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"🧪 Experiment: {EXPERIMENT_NAME}")

# ============================
# LOAD DATA
# ============================
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin-1")

if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

df["label"] = df["label"].str.lower()
df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
df.dropna(inplace=True)

X = df["message"].astype(str)
y = df["label_encoded"].astype(int)

# ============================
# SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# VECTORIZE
# ============================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ============================
# TRAIN
# ============================
with mlflow.start_run(run_name=f"spam-train-{datetime.utcnow()}"):

    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 5000)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    print("=" * 60)
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("=" * 60)

    # ============================
    # SAVE MODEL
    # ============================
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    mlflow.log_artifacts(MODEL_DIR, artifact_path="model")

    # ============================
    # QUALITY GATE
    # ============================
    ACCURACY_THRESHOLD = 0.25

    if accuracy < ACCURACY_THRESHOLD:
        print("❌ Accuracy below threshold. Aborting pipeline.")
        sys.exit(1)

    print("✅ Model passed quality gate")
