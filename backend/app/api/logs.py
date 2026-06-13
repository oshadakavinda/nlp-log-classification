import os
import shutil
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select, delete
from typing import List
from app.worker import process_csv_task, redis_client
from app.database import get_session
from fastapi import Depends
from app.models.log import LogEntry
import asyncio

router = APIRouter()

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
def get_logs(session: Session = Depends(get_session), limit: int = 100, offset: int = 0):
    logs = session.exec(select(LogEntry).offset(offset).limit(limit)).all()
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
                # If no data yet, send 0
                await websocket.send_json({"processed": 0, "total": 1})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for job {job_id}")
    finally:
        await websocket.close()
