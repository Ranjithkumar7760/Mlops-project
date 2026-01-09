"""
SMS Spam Detection - MLOps Ready Model Training Script

Features added:
- MLflow experiment tracking
- Metric & parameter logging
- Model & vectorizer artifact versioning
- Accuracy-based pipeline quality gate
- Compatible with AWS CodePipeline + EKS
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
# DIRECTORY SETUP
# ============================
MODEL_DIR = "model"
DATA_PATH = "data/spam.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# MLFLOW CONFIGURATION
# ============================
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://<MLFLOW_EC2_IP>:5000"   # Replace with MLflow EC2 IP or use CodeBuild env var
)

EXPERIMENT_NAME = "SMS-Spam-Detection"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow Experiment: {EXPERIMENT_NAME}")

# ============================
# LOAD DATA
# ============================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Handle common dataset formats
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
elif "label" not in df.columns or "message" not in df.columns:
    df.columns = ["label", "message"] + list(df.columns[2:])
    df = df[["label", "message"]]

# ============================
# DATA CLEANING & ENCODING
# ============================
print("Encoding labels...")
df["label"] = df["label"].astype(str).str.lower()
df["label_encoded"] = df["label"].map({"spam": 1, "ham": 0})

df = df.dropna(subset=["label_encoded"])

X = df["message"].astype(str)
y = df["label_encoded"].astype(int)

# ============================
# TRAIN / TEST SPLIT
# ============================
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# TF-IDF VECTORIZATION
# ============================
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================
# MLFLOW TRAINING RUN
# ============================
with mlflow.start_run(run_name=f"spam-training-{datetime.utcnow()}"):

    # Log parameters
    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("test_size", 0.2)

    # ============================
    # MODEL TRAINING
    # ============================
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    # ============================
    # EVALUATION
    # ============================
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    print("\n" + "=" * 60)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("=" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ============================
    # SAVE ARTIFACTS
    # ============================
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(vectorizer_path, artifact_path="model")

    print("Model and vectorizer saved and logged to MLflow")

    # ============================
    # QUALITY GATE (AUTO-RETRAIN CONTROL)
    # ============================
    ACCURACY_THRESHOLD = 0.90

    if accuracy < ACCURACY_THRESHOLD:
        print(
            f"❌ Training failed: Accuracy {accuracy:.4f} "
            f"is below threshold {ACCURACY_THRESHOLD}"
        )
        sys.exit(1)

    print("✅ Training successful – model meets quality threshold")
