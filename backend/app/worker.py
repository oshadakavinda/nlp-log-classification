import os
import csv
import json
import shutil
from datetime import datetime
from celery import Celery
from sqlmodel import Session, select
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import redis
import numpy as np

from app.core.config import settings
from app.database import engine
from app.models.log import LogEntry
from app.models.model_version import ModelVersion
from app.services.classify import classify_log
from app.services.processor_bert import model_embedding
from app.services.processor_regex import classify_with_regex

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

redis_client = redis.from_url(settings.CELERY_BROKER_URL)

# Common aliases for the required CSV columns.
# The first match found (in order) is used.
LOG_MESSAGE_ALIASES = [
    "log_message", "log message", "logmessage",
    "message", "msg", "log", "text", "log_text",
    "log_msg", "description", "content", "detail", "details",
]
SOURCE_ALIASES = [
    "source", "origin", "system", "application", "app",
    "service", "component", "endpoint", "module", "log_source",
    "src", "host", "hostname", "logger", "ip", "caller", "thread",
    "class", "syslog_tag", "tag", "device", "sender", "client", "agent"
]
LABEL_ALIASES = [
    "target_label", "target label", "targetlabel",
    "label", "class", "category", "type", "target", "classification",
    "prediction", "predicted", "predicted_label", "pred", "output", "target_class",
]


def _normalize_headers(fieldnames):
    """
    Build a mapping from normalized (lowercase, stripped, underscore) header
    names to the original header names as they appear in the CSV dict rows.
    """
    mapping = {}
    for name in (fieldnames or []):
        if name is None:
            continue
        clean = name.strip().strip('\ufeff').lower().replace(' ', '_')
        mapping[clean] = name
    return mapping


def _resolve_column(header_map, aliases, fallback=None):
    """
    Given a header_map {normalized_key: original_key} and a list of
    candidate aliases, return the original CSV key for the first alias that
    matches. Returns *fallback* if nothing matches.
    """
    for alias in aliases:
        normalized_alias = alias.strip().lower().replace(' ', '_')
        if normalized_alias in header_map:
            return header_map[normalized_alias]
    return fallback


@celery_app.task(bind=True)
def process_csv_task(self, file_path: str):
    total_rows = 0
    rows = []

    # Read CSV with BOM-aware encoding
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"[Worker] CSV columns detected: {fieldnames}")
        rows = list(reader)
        total_rows = len(rows)

    print(f"[Worker] Total rows to process: {total_rows}")

    if total_rows == 0:
        return {"status": "completed", "processed": 0}

    # --- Resolve which CSV columns map to 'source' and 'log_message' ---
    header_map = _normalize_headers(fieldnames)
    msg_col = _resolve_column(header_map, LOG_MESSAGE_ALIASES)
    src_col = _resolve_column(header_map, SOURCE_ALIASES)

    print(f"[Worker] Column mapping: log_message -> '{msg_col}', source -> '{src_col}'")

    if msg_col is None:
        error_msg = (
            f"[Worker] ERROR: Could not find a log message column. "
            f"CSV headers: {fieldnames}. "
            f"Expected one of: {LOG_MESSAGE_ALIASES}"
        )
        print(error_msg)
        redis_client.set(
            f"job_progress_{self.request.id}",
            json.dumps({"processed": 0, "total": total_rows, "status": "completed",
                         "error": "No log message column found in CSV"})
        )
        return {"status": "error", "error": error_msg}

    # Log the first row for debugging
    if rows:
        sample_msg = (rows[0].get(msg_col, "") or "")[:100]
        sample_src = (rows[0].get(src_col, "Unknown") if src_col else "Unknown")
        print(f"[Worker] Sample row: source='{sample_src}', log_message='{sample_msg}'")

    processed = 0
    inserted = 0
    errors = 0
    with Session(engine) as session:
        for row in rows:
            try:
                source = (row.get(src_col, "Unknown") if src_col else "Unknown") or "Unknown"
                log_msg = (row.get(msg_col, "") or "").strip()

                if not log_msg:
                    processed += 1
                    continue

                if source.strip() == "Unknown" or not source.strip():
                    from app.services.classify import infer_source
                    source = infer_source(log_msg)

                label, method, confidence = classify_log(source, log_msg)

                entry = LogEntry(
                    source=source,
                    log_message=log_msg,
                    target_label=label,
                    classification_method=method,
                    confidence=confidence
                )
                session.add(entry)
                session.commit()
                inserted += 1
            except Exception as e:
                session.rollback()
                print(f"[Worker] Error processing row {processed + 1}: {e}")
                errors += 1

            processed += 1

            # Update progress via Redis (can be polled by WebSocket)
            progress = {"processed": processed, "total": total_rows}
            redis_client.set(f"job_progress_{self.request.id}", json.dumps(progress))

    print(f"[Worker] Done. Processed={processed}, Inserted={inserted}, Errors={errors}")

    # Clean up the file
    if os.path.exists(file_path):
        os.remove(file_path)

    redis_client.set(f"job_progress_{self.request.id}", json.dumps({"processed": total_rows, "total": total_rows, "status": "completed"}))
    return {"status": "completed", "processed": processed, "inserted": inserted, "total": total_rows, "errors": errors}


@celery_app.task(bind=True)
def train_model_task(self, file_path: str, dataset_name: str):
    job_id = self.request.id
    redis_key = f"job_progress_{job_id}"
    log_key = f"job_logs_{job_id}"

    # Delete any stale logs for this job
    redis_client.delete(log_key)

    training_log_msgs = []

    def log_msg(msg):
        print(f"[Train Job {job_id}] {msg}")
        log_data = json.dumps({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        redis_client.rpush(log_key, log_data)
        # Keep logs capped at last 1000 lines
        redis_client.ltrim(log_key, -1000, -1)
        training_log_msgs.append(msg)

    version_tag = None
    accuracy = 0.0

    try:
        log_msg("=" * 60)
        log_msg("  NLP Log Classification — Model Training")
        log_msg("=" * 60)
        log_msg(f"Starting model training job: {job_id}")
        redis_client.set(redis_key, json.dumps({
            "processed": 0, "total": 100, "status": "processing", "message": "Reading training dataset..."
        }))

        # Read CSV
        rows = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        log_msg(f"\n[1/5] Loading datasets...")
        log_msg(f"  CSV columns detected: {fieldnames}")

        header_map = _normalize_headers(fieldnames)
        msg_col = _resolve_column(header_map, LOG_MESSAGE_ALIASES)
        lbl_col = _resolve_column(header_map, LABEL_ALIASES)
        src_col = _resolve_column(header_map, SOURCE_ALIASES)

        auto_labeling = False
        if msg_col is not None and lbl_col is None:
            auto_labeling = True
            log_msg("No target label column detected. Enabling dynamic pseudo-labeling via active pipeline...")
            lbl_col_display = "[Auto-labeled]"
        else:
            lbl_col_display = f"'{lbl_col}'"

        log_msg(f"  Mapped columns: log_message -> '{msg_col}', target_label -> {lbl_col_display}, source -> '{src_col}'")

        if msg_col is None or (lbl_col is None and not auto_labeling):
            error_msg = f"Failed to map columns. Required: log_message & target_label. Found: {fieldnames}"
            log_msg(f"ERROR: {error_msg}")
            redis_client.set(redis_key, json.dumps({
                "processed": 0, "total": 100, "status": "completed", "error": error_msg
            }))
            return {"status": "error", "error": error_msg}

        # Build raw dataset (all valid rows, all sources)
        raw_logs = []
        raw_labels = []
        raw_sources = []
        for row in rows:
            msg = (row.get(msg_col, "") or "").strip()
            if not msg:
                continue
            src = (row.get(src_col, "Unknown") or "Unknown").strip() if src_col else "Unknown"

            if auto_labeling:
                if src == "Unknown" or not src:
                    from app.services.classify import infer_source
                    src = infer_source(msg)
                try:
                    lbl, _, _ = classify_log(src, msg)
                except Exception:
                    lbl = "Unclassified"
            else:
                lbl = (row.get(lbl_col, "") or "").strip()

            if msg and lbl:
                raw_logs.append(msg)
                raw_labels.append(lbl)
                raw_sources.append(src)

        total_samples = len(raw_logs)
        log_msg(f"\n  Total rows loaded: {total_samples}")

        # Full label distribution
        full_distribution: dict = {}
        for lbl in raw_labels:
            full_distribution[lbl] = full_distribution.get(lbl, 0) + 1

        log_msg(f"\n  Label distribution (full dataset):")
        for lbl, cnt in sorted(full_distribution.items(), key=lambda x: -x[1]):
            log_msg(f"    {lbl}: {cnt}")

        # Step 2: Filter for BERT training (exclude regex-handled and LLM-handled rows)
        log_msg(f"\n[2/5] Filtering data for BERT processor...")
        redis_client.set(redis_key, json.dumps({
            "processed": 5, "total": 100, "status": "processing", "message": "Filtering dataset for BERT training..."
        }))

        valid_logs = []
        valid_labels = []
        excluded_regex = 0
        excluded_llm = 0

        for msg, lbl, src in zip(raw_logs, raw_labels, raw_sources):
            if src == "LegacyCRM":
                excluded_llm += 1
                continue
            if classify_with_regex(msg) is not None:
                excluded_regex += 1
                continue
            valid_logs.append(msg)
            valid_labels.append(lbl)

        log_msg(f"  Excluded {excluded_regex} regex-classifiable rows")
        log_msg(f"  Excluded {excluded_llm} LegacyCRM rows (handled by LLM)")
        log_msg(f"  Remaining rows for BERT training: {len(valid_logs)}")

        bert_distribution: dict = {}
        for lbl in valid_labels:
            bert_distribution[lbl] = bert_distribution.get(lbl, 0) + 1

        log_msg(f"\n  Label distribution (BERT training data):")
        for lbl, cnt in sorted(bert_distribution.items(), key=lambda x: -x[1]):
            log_msg(f"    {lbl}: {cnt}")

        num_records = len(valid_logs)
        log_msg(f"  Cleaned dataset: {num_records} valid training examples (skipped {total_samples - num_records} rows due to exclusions)")

        if num_records < 10:
            error_msg = f"Insufficient BERT training data. Need at least 10 valid labeled rows after filtering, found {num_records}."
            log_msg(f"ERROR: {error_msg}")
            redis_client.set(redis_key, json.dumps({
                "processed": 0, "total": 100, "status": "completed", "error": error_msg
            }))
            return {"status": "error", "error": error_msg}

        # Step 3: Encoding
        log_msg(f"\n[3/5] Generating sentence embeddings...")
        redis_client.set(redis_key, json.dumps({
            "processed": 10, "total": 100, "status": "processing", "message": "Encoding log messages using BERT..."
        }))

        embeddings = []
        batch_size = 128
        for i in range(0, num_records, batch_size):
            batch = valid_logs[i:i + batch_size]
            batch_embeddings = model_embedding.encode(batch)
            embeddings.extend(batch_embeddings)
            prog_pct = min(60, 10 + int((len(embeddings) / num_records) * 50))
            redis_client.set(redis_key, json.dumps({
                "processed": prog_pct, "total": 100, "status": "processing",
                "message": f"Encoding log messages: {len(embeddings)}/{num_records}"
            }))

        log_msg("  Sentence embeddings successfully generated.")

        # Step 4: Train/Test Split & Validation model
        X = np.array(embeddings)
        y = np.array(valid_labels)

        test_size = 0.3 if num_records >= 10 else 0.1
        log_msg(f"\n[4/5] Training model (test_size={test_size}, seed=42)...")
        redis_client.set(redis_key, json.dumps({
            "processed": 65, "total": 100, "status": "processing", "message": "Splitting data & validating model..."
        }))

        # Check if stratify is feasible (each class needs >= 2 samples)
        label_counts = {lbl: list(y).count(lbl) for lbl in set(y)}
        can_stratify = all(count >= 2 for count in label_counts.values())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if can_stratify else None
        )

        log_msg(f"  Train samples: {len(X_train)}")
        log_msg(f"  Test samples:  {len(X_test)}")

        log_msg("  Fitting Logistic Regression model (validation phase)...")
        val_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        val_clf.fit(X_train, y_train)
        y_pred = val_clf.predict(X_test)

        # Step 5: Evaluate
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix with ordered labels
        labels_ordered = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels_ordered)

        log_msg(f"\n[5/5] Evaluation Results:")
        log_msg("=" * 60)
        log_msg(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

        # Per-class report
        for lbl_key, metrics in report.items():
            if isinstance(metrics, dict):
                log_msg(
                    f"  Class '{lbl_key}': precision={metrics.get('precision', 0):.2f}, "
                    f"recall={metrics.get('recall', 0):.2f}, "
                    f"f1-score={metrics.get('f1-score', 0):.2f}, "
                    f"support={int(metrics.get('support', 0))}"
                )

        # Confusion matrix text
        log_msg(f"\n  Confusion Matrix:")
        header_str = f"  {'':20s}" + "".join(f"{l[:12]:>13s}" for l in labels_ordered)
        log_msg(header_str)
        for i, row_label in enumerate(labels_ordered):
            row_str = f"  {row_label:20s}" + "".join(f"{cm[i][j]:13d}" for j in range(len(labels_ordered)))
            log_msg(row_str)

        # Fit final model on 100% of the filtered data
        log_msg("\n  Training final Logistic Regression model on full dataset...")
        redis_client.set(redis_key, json.dumps({
            "processed": 80, "total": 100, "status": "processing", "message": "Training final model..."
        }))

        final_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        final_clf.fit(X, y)

        # Save versioned model and extended metadata
        log_msg("  Saving model weights and version metadata...")
        redis_client.set(redis_key, json.dumps({
            "processed": 90, "total": 100, "status": "processing", "message": "Saving versioned model..."
        }))

        version_tag = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        versions_dir = os.path.join(backend_dir, "models/versions")
        os.makedirs(versions_dir, exist_ok=True)

        model_path = os.path.join(versions_dir, f"{version_tag}.joblib")
        meta_path = os.path.join(versions_dir, f"{version_tag}_metadata.json")

        joblib.dump(final_clf, model_path)

        # Extended metrics data stored in metrics_json column
        metrics_data = {
            "version_tag": version_tag,
            "created_at": datetime.utcnow().isoformat(),
            "dataset_name": dataset_name,
            "accuracy": accuracy,
            "total_samples": total_samples,
            "bert_samples": num_records,
            "excluded_regex": excluded_regex,
            "excluded_llm": excluded_llm,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "labels": labels_ordered,
            "confusion_matrix": cm.tolist(),
            "full_label_distribution": full_distribution,
            "bert_label_distribution": bert_distribution,
            "classes": list(final_clf.classes_),
            "report": report,
            "training_logs": training_log_msgs,
        }

        with open(meta_path, "w") as mf:
            json.dump(metrics_data, mf)

        log_msg(f"  Model saved to: {model_path}")
        log_msg(f"  Metrics saved to: {meta_path}")

        # DB update & auto-activation
        log_msg("  Persisting model version to database and activating it...")
        redis_client.set(redis_key, json.dumps({
            "processed": 95, "total": 100, "status": "processing", "message": "Activating new model..."
        }))

        active_model_path = os.path.join(backend_dir, "models/log_classifier.joblib")
        active_meta_path = os.path.join(backend_dir, "models/log_classifier_metadata.json")

        shutil.copy2(model_path, active_model_path)
        shutil.copy2(meta_path, active_meta_path)

        with Session(engine) as session:
            stmt = select(ModelVersion).where(ModelVersion.is_active == True)
            active_versions = session.exec(stmt).all()
            for av in active_versions:
                av.is_active = False
                session.add(av)

            mv = ModelVersion(
                version_tag=version_tag,
                dataset_name=dataset_name,
                num_records=num_records,
                accuracy=accuracy,
                metrics_json=json.dumps(metrics_data),
                is_active=True
            )
            session.add(mv)
            session.commit()

        log_msg("  Model version successfully activated and saved in DB.")
        log_msg(f"\n{'=' * 60}")
        log_msg(f"  Training complete! Accuracy: {accuracy:.2%}")
        log_msg(f"{'=' * 60}")

        redis_client.set(redis_key, json.dumps({
            "processed": 100, "total": 100, "status": "completed", "message": "Model training completed successfully!"
        }))

    except Exception as e:
        import traceback
        err_tb = traceback.format_exc()
        log_msg(f"CRITICAL ERROR during training:\n{err_tb}")
        redis_client.set(redis_key, json.dumps({
            "processed": 100, "total": 100, "status": "completed", "error": str(e)
        }))
        return {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                log_msg("Temporary training file cleaned up.")
            except Exception as ex:
                log_msg(f"Warning: Failed to delete temp file {file_path}: {ex}")

    return {"status": "completed", "version_tag": version_tag, "accuracy": accuracy}


@celery_app.task(bind=True)
def train_from_db_task(self):
    job_id = self.request.id
    redis_key = f"job_progress_{job_id}"
    log_key = f"job_logs_{job_id}"

    # Delete any stale logs for this job
    redis_client.delete(log_key)

    training_log_msgs = []

    def log_msg(msg):
        print(f"[Train DB Job {job_id}] {msg}")
        log_data = json.dumps({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        redis_client.rpush(log_key, log_data)
        redis_client.ltrim(log_key, -1000, -1)
        training_log_msgs.append(msg)

    version_tag = None
    accuracy = 0.0

    try:
        log_msg("=" * 60)
        log_msg("  NLP Log Classification — DB Model Training")
        log_msg("=" * 60)
        log_msg(f"Starting model training from database logs: {job_id}")
        redis_client.set(redis_key, json.dumps({
            "processed": 0, "total": 100, "status": "processing", "message": "Fetching database logs..."
        }))

        # Fetch high-confidence or manually corrected logs from database
        with Session(engine) as session:
            db_logs = session.exec(
                select(LogEntry).where(
                    (LogEntry.user_corrected == True) | (LogEntry.confidence >= 0.8)
                )
            ).all()

        log_msg(f"\n[1/5] Loading datasets...")
        log_msg(f"  Retrieved {len(db_logs)} high-confidence or user-corrected logs from database.")

        # Build raw dataset from DB entries
        raw_logs = []
        raw_labels = []
        raw_sources = []
        for entry in db_logs:
            if entry.log_message and entry.target_label:
                raw_logs.append(entry.log_message)
                raw_labels.append(entry.target_label)
                raw_sources.append(entry.source or "Unknown")

        total_samples = len(raw_logs)
        log_msg(f"\n  Total rows loaded: {total_samples}")

        # Full label distribution
        full_distribution: dict = {}
        for lbl in raw_labels:
            full_distribution[lbl] = full_distribution.get(lbl, 0) + 1

        log_msg(f"\n  Label distribution (full dataset):")
        for lbl, cnt in sorted(full_distribution.items(), key=lambda x: -x[1]):
            log_msg(f"    {lbl}: {cnt}")

        # Step 2: Filter for BERT training
        log_msg(f"\n[2/5] Filtering data for BERT processor...")
        redis_client.set(redis_key, json.dumps({
            "processed": 5, "total": 100, "status": "processing", "message": "Filtering dataset for BERT training..."
        }))

        valid_logs = []
        valid_labels = []
        excluded_regex = 0
        excluded_llm = 0

        for msg, lbl, src in zip(raw_logs, raw_labels, raw_sources):
            if src == "LegacyCRM":
                excluded_llm += 1
                continue
            if classify_with_regex(msg) is not None:
                excluded_regex += 1
                continue
            valid_logs.append(msg)
            valid_labels.append(lbl)

        log_msg(f"  Excluded {excluded_regex} regex-classifiable rows")
        log_msg(f"  Excluded {excluded_llm} LegacyCRM rows (handled by LLM)")
        log_msg(f"  Remaining rows for BERT training: {len(valid_logs)}")

        bert_distribution: dict = {}
        for lbl in valid_labels:
            bert_distribution[lbl] = bert_distribution.get(lbl, 0) + 1

        log_msg(f"\n  Label distribution (BERT training data):")
        for lbl, cnt in sorted(bert_distribution.items(), key=lambda x: -x[1]):
            log_msg(f"    {lbl}: {cnt}")

        num_records = len(valid_logs)
        log_msg(f"  Prepared dataset: {num_records} valid training examples.")

        if num_records < 10:
            error_msg = f"Insufficient BERT training data in DB. Need at least 10 high-confidence logs after filtering, found {num_records}."
            log_msg(f"ERROR: {error_msg}")
            redis_client.set(redis_key, json.dumps({
                "processed": 0, "total": 100, "status": "completed", "error": error_msg
            }))
            return {"status": "error", "error": error_msg}

        # Step 3: Encoding
        log_msg(f"\n[3/5] Generating sentence embeddings...")
        redis_client.set(redis_key, json.dumps({
            "processed": 10, "total": 100, "status": "processing", "message": "Encoding log messages using BERT..."
        }))

        embeddings = []
        batch_size = 128
        for i in range(0, num_records, batch_size):
            batch = valid_logs[i:i + batch_size]
            batch_embeddings = model_embedding.encode(batch)
            embeddings.extend(batch_embeddings)
            prog_pct = min(60, 10 + int((len(embeddings) / num_records) * 50))
            redis_client.set(redis_key, json.dumps({
                "processed": prog_pct, "total": 100, "status": "processing",
                "message": f"Encoding log messages: {len(embeddings)}/{num_records}"
            }))

        log_msg("  Sentence embeddings successfully generated.")

        # Step 4: Train/Test Split & Validation
        X = np.array(embeddings)
        y = np.array(valid_labels)

        test_size = 0.3 if num_records >= 10 else 0.1
        log_msg(f"\n[4/5] Training model (test_size={test_size}, seed=42)...")
        redis_client.set(redis_key, json.dumps({
            "processed": 65, "total": 100, "status": "processing", "message": "Splitting data & validating model..."
        }))

        # Check if stratify is feasible (each class needs >= 2 samples)
        label_counts = {lbl: list(y).count(lbl) for lbl in set(y)}
        can_stratify = all(count >= 2 for count in label_counts.values())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if can_stratify else None
        )

        log_msg(f"  Train samples: {len(X_train)}")
        log_msg(f"  Test samples:  {len(X_test)}")

        log_msg("  Fitting Logistic Regression model (validation phase)...")
        val_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        val_clf.fit(X_train, y_train)
        y_pred = val_clf.predict(X_test)

        # Step 5: Evaluate
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        labels_ordered = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels_ordered)

        log_msg(f"\n[5/5] Evaluation Results:")
        log_msg("=" * 60)
        log_msg(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

        for lbl_key, metrics in report.items():
            if isinstance(metrics, dict):
                log_msg(
                    f"  Class '{lbl_key}': precision={metrics.get('precision', 0):.2f}, "
                    f"recall={metrics.get('recall', 0):.2f}, "
                    f"f1-score={metrics.get('f1-score', 0):.2f}, "
                    f"support={int(metrics.get('support', 0))}"
                )

        log_msg(f"\n  Confusion Matrix:")
        header_str = f"  {'':20s}" + "".join(f"{l[:12]:>13s}" for l in labels_ordered)
        log_msg(header_str)
        for i, row_label in enumerate(labels_ordered):
            row_str = f"  {row_label:20s}" + "".join(f"{cm[i][j]:13d}" for j in range(len(labels_ordered)))
            log_msg(row_str)

        # Final model on 100% of data
        log_msg("\n  Training final Logistic Regression model on full dataset...")
        redis_client.set(redis_key, json.dumps({
            "processed": 80, "total": 100, "status": "processing", "message": "Training final model..."
        }))

        final_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        final_clf.fit(X, y)

        # Save versioned model and extended metadata
        log_msg("  Saving model weights and version metadata...")
        redis_client.set(redis_key, json.dumps({
            "processed": 90, "total": 100, "status": "processing", "message": "Saving versioned model..."
        }))

        version_tag = f"v_db_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        versions_dir = os.path.join(backend_dir, "models/versions")
        os.makedirs(versions_dir, exist_ok=True)

        model_path = os.path.join(versions_dir, f"{version_tag}.joblib")
        meta_path = os.path.join(versions_dir, f"{version_tag}_metadata.json")

        joblib.dump(final_clf, model_path)

        # Extended metrics data
        metrics_data = {
            "version_tag": version_tag,
            "created_at": datetime.utcnow().isoformat(),
            "dataset_name": "Database Logs",
            "accuracy": accuracy,
            "total_samples": total_samples,
            "bert_samples": num_records,
            "excluded_regex": excluded_regex,
            "excluded_llm": excluded_llm,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "labels": labels_ordered,
            "confusion_matrix": cm.tolist(),
            "full_label_distribution": full_distribution,
            "bert_label_distribution": bert_distribution,
            "classes": list(final_clf.classes_),
            "report": report,
            "training_logs": training_log_msgs,
        }

        with open(meta_path, "w") as mf:
            json.dump(metrics_data, mf)

        log_msg(f"  Model saved to: {model_path}")
        log_msg(f"  Metrics saved to: {meta_path}")

        log_msg("  Persisting model version and activating it...")
        redis_client.set(redis_key, json.dumps({
            "processed": 95, "total": 100, "status": "processing", "message": "Activating new model..."
        }))

        active_model_path = os.path.join(backend_dir, "models/log_classifier.joblib")
        active_meta_path = os.path.join(backend_dir, "models/log_classifier_metadata.json")

        shutil.copy2(model_path, active_model_path)
        shutil.copy2(meta_path, active_meta_path)

        with Session(engine) as session:
            stmt = select(ModelVersion).where(ModelVersion.is_active == True)
            active_versions = session.exec(stmt).all()
            for av in active_versions:
                av.is_active = False
                session.add(av)

            mv = ModelVersion(
                version_tag=version_tag,
                dataset_name="Database Logs",
                num_records=num_records,
                accuracy=accuracy,
                metrics_json=json.dumps(metrics_data),
                is_active=True
            )
            session.add(mv)
            session.commit()

        log_msg("  Model version successfully activated and saved in DB.")
        log_msg(f"\n{'=' * 60}")
        log_msg(f"  Training complete! Accuracy: {accuracy:.2%}")
        log_msg(f"{'=' * 60}")

        redis_client.set(redis_key, json.dumps({
            "processed": 100, "total": 100, "status": "completed", "message": "Model training completed successfully!"
        }))

    except Exception as e:
        import traceback
        err_tb = traceback.format_exc()
        log_msg(f"CRITICAL ERROR during DB training:\n{err_tb}")
        redis_client.set(redis_key, json.dumps({
            "processed": 100, "total": 100, "status": "completed", "error": str(e)
        }))
        return {"status": "error", "error": str(e)}

    return {"status": "completed", "version_tag": version_tag, "accuracy": accuracy}
