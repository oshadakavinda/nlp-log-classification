import os
import shutil
import json
from datetime import datetime
from typing import List, Optional, Union
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select, delete
from app.worker import process_csv_task, train_model_task, redis_client, celery_app
from app.database import get_session
from app.models.log import LogEntry
from app.models.model_version import ModelVersion
from app.models.regex_rule import RegexRule
from app.services.classify import classify_log, infer_source
from app.core.config import settings
import asyncio
import re

router = APIRouter()

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class SingleLogRequest(BaseModel):
    log_message: str
    source: Optional[str] = "Unknown"


class LogClassifyResponse(BaseModel):
    log_message: str
    source: str
    target_label: str
    classification_method: str
    confidence: float


@router.post("/upload")
async def upload_logs(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    task = process_csv_task.delay(file_path)
    return {"job_id": task.id, "message": "CSV upload processing started."}


@router.get("", response_model=List[LogEntry])
def get_logs(session: Session = Depends(get_session), limit: int = 500, offset: int = 0):
    session.expire_all()
    logs = session.exec(select(LogEntry).order_by(LogEntry.created_at.desc()).offset(offset).limit(limit)).all()
    return logs


@router.delete("")
def delete_logs(session: Session = Depends(get_session)):
    session.exec(delete(LogEntry))
    session.commit()
    return {"message": "All logs deleted successfully"}


@router.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            progress_data = redis_client.get(f"job_progress_{job_id}")
            if progress_data:
                data = json.loads(progress_data)
                await websocket.send_json(data)
                if data.get("status") == "completed":
                    break
            else:
                await websocket.send_json({"processed": 0, "total": 1, "status": "processing"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --- Model Training & Versioning Endpoints ---

@router.post("/train")
async def train_model(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
        
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    upload_dir = os.path.join(backend_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    task = train_model_task.delay(file_path, file.filename)
    return {"job_id": task.id, "message": "Model training task started."}


@router.get("/train/logs/{job_id}")
def get_training_logs(job_id: str):
    log_key = f"job_logs_{job_id}"
    logs = redis_client.lrange(log_key, 0, -1)
    parsed_logs = []
    for log in logs:
        try:
            parsed_logs.append(json.loads(log))
        except Exception:
            parsed_logs.append({"timestamp": datetime.utcnow().isoformat(), "message": log.decode("utf-8")})
    return parsed_logs


@router.get("/model-versions", response_model=List[ModelVersion])
def get_model_versions(session: Session = Depends(get_session)):
    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    return session.exec(stmt).all()


@router.post("/model-versions/{version_id}/activate")
def activate_model_version(version_id: int, session: Session = Depends(get_session)):
    mv = session.get(ModelVersion, version_id)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found.")
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(backend_dir, f"models/versions/{mv.version_tag}.joblib")
    meta_path = os.path.join(backend_dir, f"models/versions/{mv.version_tag}_metadata.json")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model files for version tag {mv.version_tag} do not exist on disk.")
        
    active_model_path = os.path.join(backend_dir, "models/log_classifier.joblib")
    active_meta_path = os.path.join(backend_dir, "models/log_classifier_metadata.json")
    
    try:
        shutil.copy2(model_path, active_model_path)
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, active_meta_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy model files: {e}")
        
    stmt = select(ModelVersion).where(ModelVersion.is_active == True)
    active_versions = session.exec(stmt).all()
    for av in active_versions:
        av.is_active = False
        session.add(av)
        
    mv.is_active = True
    session.add(mv)
    session.commit()
    
    return {"status": "success", "message": f"Successfully activated model version {mv.version_tag}."}


@router.get("/active-model")
def get_active_model(session: Session = Depends(get_session)):
    stmt = select(ModelVersion).where(ModelVersion.is_active == True)
    active_version = session.exec(stmt).first()
    
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    active_meta_path = os.path.join(backend_dir, "models/log_classifier_metadata.json")
    
    file_meta = {}
    if os.path.exists(active_meta_path):
        try:
            with open(active_meta_path, "r") as f:
                file_meta = json.load(f)
        except Exception:
            pass
            
    if active_version:
        return {
            "version_tag": active_version.version_tag,
            "dataset_name": active_version.dataset_name,
            "num_records": active_version.num_records,
            "accuracy": active_version.accuracy,
            "created_at": active_version.created_at.isoformat(),
            "report": json.loads(active_version.metrics_json),
            "classes": file_meta.get("classes", [])
        }
    elif file_meta:
        return file_meta
    else:
        return {"status": "no_model_loaded", "message": "No active model version registered."}


@router.get("/download-dataset")
def download_dataset():
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(backend_dir, "training/dataset/synthetic_logs.csv")
    if os.path.exists(dataset_path):
        return FileResponse(dataset_path, media_type="text/csv", filename="synthetic_logs.csv")
    else:
        raise HTTPException(status_code=404, detail="Sample training dataset not found.")


@router.get("/system-status")
def get_system_status(session: Session = Depends(get_session)):
    db_ok = False
    try:
        session.exec(select(1))
        db_ok = True
    except Exception:
        pass

    redis_ok = False
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        pass

    worker_ok = False
    try:
        i = celery_app.control.inspect(timeout=0.5)
        ping_res = i.ping()
        if ping_res and len(ping_res) > 0:
            worker_ok = True
    except Exception:
        pass

    llm_configured = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())

    return {
        "database": "healthy" if db_ok else "unhealthy",
        "redis": "healthy" if redis_ok else "unhealthy",
        "celery_worker": "active" if worker_ok else "inactive",
        "groq_api": "configured" if llm_configured else "not_configured"
    }


# --- Production-Ready Real-Time Classification Service API ---

@router.post("/classify")
def classify_logs_api(
    payload: Union[SingleLogRequest, List[SingleLogRequest]],
    save_to_db: bool = False,
    session: Session = Depends(get_session)
):
    is_list = isinstance(payload, list)
    items = payload if is_list else [payload]
    
    results = []
    for item in items:
        source = (item.source or "Unknown").strip()
        if source == "Unknown" or not source:
            source = infer_source(item.log_message)
            
        label, method, confidence = classify_log(source, item.log_message)
        
        entry_data = {
            "log_message": item.log_message,
            "source": source,
            "target_label": label,
            "classification_method": method,
            "confidence": round(confidence, 4)
        }
        
        if save_to_db:
            db_entry = LogEntry(**entry_data)
            session.add(db_entry)
            
        results.append(entry_data)
        
    if save_to_db:
        session.commit()
        
    return results if is_list else results[0]


# --- Dynamic Regex Rules CRUD Endpoints ---

class RegexRuleCreate(BaseModel):
    pattern: str
    target_label: str


@router.get("/regex-rules", response_model=List[RegexRule])
def get_regex_rules_api(session: Session = Depends(get_session)):
    return session.exec(select(RegexRule)).all()


@router.post("/regex-rules", response_model=RegexRule)
def create_regex_rule_api(payload: RegexRuleCreate, session: Session = Depends(get_session)):
    try:
        re.compile(payload.pattern)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid regular expression pattern: {e}")
        
    rule = RegexRule(pattern=payload.pattern, target_label=payload.target_label)
    session.add(rule)
    session.commit()
    session.refresh(rule)
    return rule


@router.delete("/regex-rules/{rule_id}")
def delete_regex_rule_api(rule_id: int, session: Session = Depends(get_session)):
    rule = session.get(RegexRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Regex rule not found.")
    session.delete(rule)
    session.commit()
    return {"status": "success", "message": f"Successfully deleted regex rule with ID {rule_id}."}
