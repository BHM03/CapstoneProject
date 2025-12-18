# train_classifier.py
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_all_processed(processed_dir: Path) -> pd.DataFrame:
    
    #Load all normalized and labeled transaction CSVs from the given directory
    files = sorted(processed_dir.glob("transactions_*.csv"))
    if not files:
        raise RuntimeError(f"No processed CSVs found in {processed_dir}")
    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Train and evaluate spending category classifier."
    )
    # default dir is now data/training instead of data/processed
    ap.add_argument(
        "--processed-dir",
        default="data/training",
        help="Directory with training CSVs (transactions_*.csv). Default: data/training",
    )
    ap.add_argument(
        "--models-dir",
        default="data/models",
        help="Where to save the trained model.",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_processed(processed_dir)

    # only use labeled rows
    df = df.dropna(subset=["category"]).copy()
    if df.empty:
        raise RuntimeError(
            "No labeled transactions found in training data. "
            "Make sure your CSVs have a 'category' column."
        )

    df["category"] = df["category"].astype(str)

    X = df["merchant_raw"]
    y = df["category"]

    # Handle tiny classes safely
    class_counts = y.value_counts()
    min_count = class_counts.min()

    if min_count < 2 or len(class_counts) == 1:
        print("[WARN] Some categories have fewer than 2 samples or only one category present.")
        print("[WARN] Current class counts:")
        print(class_counts)
        print(
            "[WARN] Falling back to NON-stratified train/test split.\n"
            "       For better evaluation, try labeling at least 2 samples per category."
        )
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=stratify_arg,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    model_path = models_dir / "category_classifier.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n[TRAIN] Saved classifier to {model_path}")


if __name__ == "__main__":
    main()
