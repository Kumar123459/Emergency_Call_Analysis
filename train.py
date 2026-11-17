
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "emergency_sentiment.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"


def build_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    return model


def train_and_save_model():
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna()

    X = data["call_text"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    return model


if __name__ == "__main__":
    train_and_save_model()
