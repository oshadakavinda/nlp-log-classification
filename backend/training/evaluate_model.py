"""
Evaluate a saved log classification model against test data.

This script loads a saved .joblib model and evaluates it against a CSV dataset,
producing detailed accuracy metrics and a confusion matrix visualization.

Usage:
    python training/evaluate_model.py --model models/log_classifier.joblib --data training/dataset/synthetic_logs.csv
    python training/evaluate_model.py --model models/log_classifier.joblib --data-dir training/dataset
"""

import argparse
import glob
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Regex patterns (same as processor_regex.py)
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
    patterns = get_regex_patterns()
    for pattern, label in patterns.items():
        try:
            if re.search(pattern, log_message):
                return label
        except Exception:
            continue
    return None


def load_data(data_path: str = None, data_dir: str = None) -> pd.DataFrame:
    """Load data from a single CSV or a directory of CSVs."""
    if data_path:
        print(f"  Loading: {data_path}")
        df = pd.read_csv(data_path)
        return df
    elif data_dir:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            print(f"ERROR: No CSV files found in {data_dir}")
            sys.exit(1)
        frames = []
        for f in csv_files:
            print(f"  Loading: {f}")
            frames.append(pd.read_csv(f))
        return pd.concat(frames, ignore_index=True)
    else:
        print("ERROR: Provide --data or --data-dir")
        sys.exit(1)


def evaluate(
    model_path: str,
    data_path: str = None,
    data_dir: str = None,
    output_dir: str = None,
):
    """Evaluate a saved model."""
    print("=" * 60)
    print("  NLP Log Classification — Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n[1/4] Loading model: {model_path}")
    clf = joblib.load(model_path)
    print(f"  Model classes: {list(clf.classes_)}")

    # Load data
    print("\n[2/4] Loading evaluation data...")
    df = load_data(data_path, data_dir)

    required_cols = {"source", "log_message", "target_label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    df = df.dropna(subset=["source", "log_message", "target_label"])

    # Filter for BERT-eligible rows only
    df["regex_label"] = df["log_message"].apply(classify_with_regex)
    df_eval = df[df["regex_label"].isna()].copy()
    df_eval = df_eval[df_eval["source"] != "LegacyCRM"]
    df_eval = df_eval.drop(columns=["regex_label"])

    print(f"  Total rows: {len(df)}")
    print(f"  BERT-eligible rows for evaluation: {len(df_eval)}")

    if len(df_eval) == 0:
        print("ERROR: No BERT-eligible rows to evaluate.")
        sys.exit(1)

    # Generate embeddings
    print("\n[3/4] Generating sentence embeddings...")
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    X = model_embedding.encode(
        df_eval["log_message"].tolist(), show_progress_bar=True
    )
    y_true = df_eval["target_label"].values

    # Predict
    print("\n[4/4] Running predictions & evaluation...")
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    accuracy = accuracy_score(y_true, y_pred)
    report_str = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))

    print("\n" + "=" * 60)
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)
    print(report_str)

    # Confusion matrix text
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

    # Show low-confidence predictions
    max_proba = y_proba.max(axis=1)
    low_conf_mask = max_proba < 0.7
    low_conf_count = low_conf_mask.sum()
    if low_conf_count > 0:
        print(f"\n  WARNING: {low_conf_count} predictions with confidence < 70%:")
        low_conf_indices = np.where(low_conf_mask)[0]
        for idx in low_conf_indices[:10]:  # Show first 10
            msg = df_eval.iloc[idx]["log_message"][:80]
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            conf = max_proba[idx]
            status = "[OK]" if true_label == pred_label else "[FAIL]"
            print(f"    {status} [{conf:.2f}] True: {true_label}, Pred: {pred_label}")
            print(f"      \"{msg}...\"")

    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        eval_metrics = {
            "accuracy": float(accuracy),
            "total_evaluated": len(df_eval),
            "labels": labels,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "low_confidence_count": int(low_conf_count),
        }
        metrics_path = os.path.join(output_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\n  Evaluation metrics saved to: {metrics_path}")

        # Confusion matrix plot
        try:
            import matplotlib
            matplotlib.use("Agg")
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
            ax.set_title(f"Evaluation Confusion Matrix (Accuracy: {accuracy:.2%})")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "eval_confusion_matrix.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Confusion matrix plot saved to: {plot_path}")
        except ImportError:
            print("  (matplotlib/seaborn not installed — skipping plot)")

    print("\n" + "=" * 60)
    print(f"  Evaluation complete! Accuracy: {accuracy:.2%}")
    print("=" * 60)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved log classification model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/log_classifier.joblib",
        help="Path to the saved .joblib model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a single CSV file for evaluation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing CSV files for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save evaluation outputs (default: models)",
    )
    args = parser.parse_args()

    if not args.data and not args.data_dir:
        args.data_dir = "training/dataset"

    evaluate(
        model_path=args.model,
        data_path=args.data,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
