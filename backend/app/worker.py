import os
import csv
import json
import shutil
from datetime import datetime
from celery import Celery
from sqlmodel import Session, select
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import redis

from app.core.config import settings
from app.database import engine
from app.models.log import LogEntry
from app.models.model_version import ModelVersion
from app.services.classify import classify_log
from app.services.processor_bert import model_embedding

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
    
    def log_msg(msg):
        print(f"[Train Job {job_id}] {msg}")
        log_data = json.dumps({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        redis_client.rpush(log_key, log_data)
        # Keep logs capped at last 1000 lines
        redis_client.ltrim(log_key, -1000, -1)
    
    try:
        log_msg(f"Starting model training job: {job_id}")
        redis_client.set(redis_key, json.dumps({"processed": 0, "total": 100, "status": "processing", "message": "Reading training dataset..."}))
        
        # Read CSV
        rows = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        
        log_msg(f"CSV columns detected: {fieldnames}")
        log_msg(f"Total rows in uploaded CSV: {len(rows)}")
        
        header_map = _normalize_headers(fieldnames)
        msg_col = _resolve_column(header_map, LOG_MESSAGE_ALIASES)
        lbl_col = _resolve_column(header_map, LABEL_ALIASES)
        src_col = _resolve_column(header_map, SOURCE_ALIASES)
        
        auto_labeling = False
        if msg_col is not None and lbl_col is None:
            auto_labeling = True
            log_msg("No target label column detected in CSV. Enabling dynamic pseudo-labeling using the active pipeline (Regex + ML + LLM fallback)...")
            lbl_col_display = "[Auto-labeled]"
        else:
            lbl_col_display = f"'{lbl_col}'"
            
        log_msg(f"Mapped columns: log_message -> '{msg_col}', target_label -> {lbl_col_display}")
        
        if msg_col is None or (lbl_col is None and not auto_labeling):
            error_msg = f"Failed to map columns. Required: log_message & target_label. Found: {fieldnames}"
            log_msg(f"ERROR: {error_msg}")
            redis_client.set(redis_key, json.dumps({
                "processed": 0, "total": 100, "status": "completed", 
                "error": error_msg
            }))
            return {"status": "error", "error": error_msg}
        
        # Prepare dataset
        valid_logs = []
        valid_labels = []
        for idx, row in enumerate(rows):
            msg = (row.get(msg_col, "") or "").strip()
            if not msg:
                continue
                
            if auto_labeling:
                src = (row.get(src_col, "Unknown") or "Unknown").strip() if src_col else "Unknown"
                try:
                    lbl, _, _ = classify_log(src, msg)
                except Exception as e:
                    lbl = "Unclassified"
            else:
                lbl = (row.get(lbl_col, "") or "").strip()
                
            if msg and lbl:
                valid_logs.append(msg)
                valid_labels.append(lbl)
        
        num_records = len(valid_logs)
        log_msg(f"Cleaned dataset: {num_records} valid training examples (skipped {len(rows) - num_records} rows due to empty values)")
        
        if num_records < 10:
            error_msg = f"Insufficient training data. Need at least 10 valid labeled rows, found {num_records}."
            log_msg(f"ERROR: {error_msg}")
            redis_client.set(redis_key, json.dumps({
                "processed": 0, "total": 100, "status": "completed", 
                "error": error_msg
            }))
            return {"status": "error", "error": error_msg}
        
        # 1. Encoding
        log_msg("Encoding log messages using SentenceTransformer (all-MiniLM-L6-v2)...")
        redis_client.set(redis_key, json.dumps({"processed": 10, "total": 100, "status": "processing", "message": "Encoding log messages using BERT..."}))
        
        # Encode in batches and report progress
        embeddings = []
        batch_size = 128
        for i in range(0, num_records, batch_size):
            batch = valid_logs[i:i+batch_size]
            batch_embeddings = model_embedding.encode(batch)
            embeddings.extend(batch_embeddings)
            prog_pct = min(60, 10 + int((len(embeddings) / num_records) * 50))
            redis_client.set(redis_key, json.dumps({
                "processed": prog_pct, "total": 100, "status": "processing", 
                "message": f"Encoding log messages: {len(embeddings)}/{num_records}"
            }))
        
        log_msg("Sentence embeddings successfully generated.")
        
        # 2. Train/Test Split & Validation
        import numpy as np
        X = np.array(embeddings)
        y = np.array(valid_labels)
        
        log_msg("Splitting dataset into 70% train and 30% test...")
        redis_client.set(redis_key, json.dumps({"processed": 65, "total": 100, "status": "processing", "message": "Splitting data & validating model..."}))
        
        test_size = 0.3 if num_records >= 10 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        log_msg(f"Train set: {len(X_train)} samples. Test set: {len(X_test)} samples.")
        log_msg("Fitting Logistic Regression model (validation phase)...")
        
        val_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        val_clf.fit(X_train, y_train)
        y_pred = val_clf.predict(X_test)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        log_msg(f"Validation Accuracy: {accuracy:.4f}")
        log_msg("Detailed Validation Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                log_msg(f"  Class '{label}': precision={metrics.get('precision', 0):.2f}, recall={metrics.get('recall', 0):.2f}, f1-score={metrics.get('f1-score', 0):.2f}")
        
        # 3. Fit Final Model on 100% of data
        log_msg("Training final Logistic Regression model on full dataset...")
        redis_client.set(redis_key, json.dumps({"processed": 80, "total": 100, "status": "processing", "message": "Training final model..."}))
        
        final_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        final_clf.fit(X, y)
        
        # 4. Save Versioned Model and Metadata
        log_msg("Saving model weights and version metadata...")
        redis_client.set(redis_key, json.dumps({"processed": 90, "total": 100, "status": "processing", "message": "Saving versioned model..."}))
        
        version_tag = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        versions_dir = os.path.join(backend_dir, "models/versions")
        os.makedirs(versions_dir, exist_ok=True)
        
        model_path = os.path.join(versions_dir, f"{version_tag}.joblib")
        meta_path = os.path.join(versions_dir, f"{version_tag}_metadata.json")
        
        joblib.dump(final_clf, model_path)
        
        # Format metadata
        metadata = {
            "version_tag": version_tag,
            "created_at": datetime.utcnow().isoformat(),
            "dataset_name": dataset_name,
            "num_records": num_records,
            "accuracy": accuracy,
            "classes": list(final_clf.classes_),
            "report": report
        }
        with open(meta_path, "w") as mf:
            json.dump(metadata, mf)
        
        log_msg(f"Saved versioned model to {model_path}")
        
        # 5. DB update & Auto-activation
        log_msg("Persisting model version to PostgreSQL and activating it...")
        redis_client.set(redis_key, json.dumps({"processed": 95, "total": 100, "status": "processing", "message": "Activating new model..."}))
        
        # Copy to active paths
        active_model_path = os.path.join(backend_dir, "models/log_classifier.joblib")
        active_meta_path = os.path.join(backend_dir, "models/log_classifier_metadata.json")
        
        shutil.copy2(model_path, active_model_path)
        shutil.copy2(meta_path, active_meta_path)
        
        with Session(engine) as session:
            # Set other versions inactive
            stmt = select(ModelVersion).where(ModelVersion.is_active == True)
            active_versions = session.exec(stmt).all()
            for av in active_versions:
                av.is_active = False
                session.add(av)
            
            # Add new active version
            mv = ModelVersion(
                version_tag=version_tag,
                dataset_name=dataset_name,
                num_records=num_records,
                accuracy=accuracy,
                metrics_json=json.dumps(report),
                is_active=True
            )
            session.add(mv)
            session.commit()
        
        log_msg("Model version successfully activated and saved in DB.")
        redis_client.set(redis_key, json.dumps({"processed": 100, "total": 100, "status": "completed", "message": "Model training completed successfully!"}))
        
    except Exception as e:
        import traceback
        err_tb = traceback.format_exc()
        log_msg(f"CRITICAL ERROR during training:\n{err_tb}")
        redis_client.set(redis_key, json.dumps({
            "processed": 100, "total": 100, "status": "completed", 
            "error": str(e)
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
