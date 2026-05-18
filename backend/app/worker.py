import os
import csv
from celery import Celery
from app.core.config import settings
from app.database import get_session, engine
from sqlmodel import Session
from app.models.log import LogEntry
from app.services.classify import classify_log
import redis
import json

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

redis_client = redis.from_url(settings.CELERY_BROKER_URL)

@celery_app.task(bind=True)
def process_csv_task(self, file_path: str):
    total_rows = 0
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total_rows = len(rows)

    if total_rows == 0:
        return {"status": "completed", "processed": 0}

    processed = 0
    with Session(engine) as session:
        for row in rows:
            source = row.get("source", "Unknown")
            log_msg = row.get("log_message", "")
            
            if not log_msg:
                processed += 1
                continue

            label, method = classify_log(source, log_msg)

            entry = LogEntry(
                source=source,
                log_message=log_msg,
                target_label=label,
                classification_method=method
            )
            session.add(entry)
            session.commit()
            
            processed += 1
            
            # Update progress via Redis (can be polled by WebSocket)
            progress = {"processed": processed, "total": total_rows}
            redis_client.set(f"job_progress_{self.request.id}", json.dumps(progress))

    # Clean up the file
    if os.path.exists(file_path):
        os.remove(file_path)

    redis_client.set(f"job_progress_{self.request.id}", json.dumps({"processed": total_rows, "total": total_rows, "status": "completed"}))
    return {"status": "completed", "processed": processed, "total": total_rows}
