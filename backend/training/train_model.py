"""
Train the log classification model.

This script:
1. Loads training data from one or more CSV files
2. Filters out regex-classifiable and LegacyCRM rows (handled by other processors)
3. Generates sentence embeddings using all-MiniLM-L6-v2
4. Trains a LogisticRegression classifier
5. Evaluates on a held-out test set
6. Saves the model + metrics

Usage (local):
    python training/train_model.py --data-dir training/dataset --output-dir models

Usage (with new data):
    1. Add your new CSV files to training/dataset/
    2. Run: python training/train_model.py --data-dir training/dataset --output-dir models

CSV files must have columns: source, log_message, target_label
"""

import argparse
import json
import os
import re
import sys
import glob
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Regex patterns (same as processor_regex.py) — these rows are excluded from
# BERT training because they are handled by the regex processor at inference.
# ---------------------------------------------------------------------------
REGEX_PATTERNS = {
    r"User User\d+ logged (in|out).": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    r"Backup completed successfully.": "System Notification",
    r"System updated to version .*": "System Notification",
    r"File .* uploaded successfully by user .*": "System Notification",
    r"Disk cleanup completed successfully.": "System Notification",
    r"System reboot initiated by user .*": "System Notification",
    r"Account with ID .* created by .*": "User Action",
}


LOADED_REGEX_PATTERNS = None

def get_regex_patterns() -> dict:
    global LOADED_REGEX_PATTERNS
    if LOADED_REGEX_PATTERNS is not None:
        return LOADED_REGEX_PATTERNS
        
    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_dir not in sys.path:
            sys.path.append(backend_dir)
            
        from sqlmodel import Session, select
        from app.database import engine
        from app.models.regex_rule import RegexRule
        
        with Session(engine) as session:
            db_rules = session.exec(select(RegexRule)).all()
            if db_rules:
                print(f"  Successfully loaded {len(db_rules)} dynamic regex rules from the database.")
                LOADED_REGEX_PATTERNS = {rule.pattern: rule.target_label for rule in db_rules}
                return LOADED_REGEX_PATTERNS
    except Exception as e:
        print(f"  Warning: Could not connect to DB to load dynamic regex rules ({e}). Using hardcoded fallback.")
        
    LOADED_REGEX_PATTERNS = REGEX_PATTERNS
    return LOADED_REGEX_PATTERNS


def classify_with_regex(log_message: str) -> str | None:
    """Return a label if the log matches a regex pattern, else None."""
    patterns = get_regex_patterns()
    for pattern, label in patterns.items():
        try:
            if re.search(pattern, log_message):
                return label
        except Exception:
            continue
    return None


def load_datasets(data_dir: str) -> pd.DataFrame:
    """Load and merge all CSV files from the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        sys.exit(1)

    frames = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        # Validate required columns
        required_cols = {"source", "log_message", "target_label"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"  WARNING: {csv_file} missing columns {missing}, skipping.")
            continue
        frames.append(df)

    if not frames:
        print("ERROR: No valid CSV files with required columns found.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    # Drop rows with NaN in critical columns
    combined = combined.dropna(subset=["source", "log_message", "target_label"])
    print(f"\n  Total rows loaded: {len(combined)}")
    return combined



def train(
    data_dir: str,
    output_dir: str,
    test_size: float = 0.3,
    random_seed: int = 42,
    max_iter: int = 1000,
):
    """Main training pipeline."""
    print("=" * 60)
    print("  NLP Log Classification — Model Training")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading datasets...")
    df = load_datasets(data_dir)

    # Show label distribution
    print("\n  Label distribution (full dataset):")
    for label, count in df["target_label"].value_counts().items():
        print(f"    {label}: {count}")

    # Step 2: Ensure enough data
    if len(df) < 10:
        print("ERROR: Not enough data for training.")
        sys.exit(1)

    # Step 3: Generate embeddings
    print("\n[2/5] Generating sentence embeddings...")
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    X = model_embedding.encode(
        df["log_message"].tolist(), show_progress_bar=True
    )
    y = df["target_label"].values

    # Step 4: Train/test split + Logistic Regression
    print(f"\n[4/5] Training model (test_size={test_size}, seed={random_seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")

    clf = LogisticRegression(max_iter=max_iter, random_state=random_seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Step 5: Evaluate
    print("\n[5/5] Evaluation Results:")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))

    print(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print(report_str)

    # Print confusion matrix as text
    print("\n  Confusion Matrix:")
    print(f"  {'':20s}", end="")
    for l in labels:
        print(f"{l[:12]:>13s}", end="")
    print()
    for i, row_label in enumerate(labels):
        print(f"  {row_label:20s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:13d}", end="")
        print()

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "log_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"\n  Model saved to: {model_path}")

    # Save metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "test_size": test_size,
        "random_seed": random_seed,
        "total_samples": len(df),
        "bert_samples": len(df_bert),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "labels": labels,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Generate confusion matrix plot
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix plot saved to: {plot_path}")
    except ImportError:
        print("  (matplotlib/seaborn not installed — skipping plot)")

    print("\n" + "=" * 60)
    print(f"  Training complete! Accuracy: {accuracy:.2%}")
    print("=" * 60)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the NLP log classification model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training/dataset",
        help="Directory containing CSV training data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained model and metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data to use for testing (default: 0.3)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for LogisticRegression (default: 1000)",
    )
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
    )
