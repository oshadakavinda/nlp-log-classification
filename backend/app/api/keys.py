import os
import secrets
import hashlib
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlmodel import Session, select
from app.database import get_session
from app.models.api_key import APIKey

router = APIRouter()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class APIKeyCreate(BaseModel):
    name: str

class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    created_at: datetime
    is_active: bool
    total_calls: int

class APIKeyCreateResponse(BaseModel):
    id: int
    name: str
    raw_key: str
    key_prefix: str
    created_at: datetime

def get_api_key(
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
    session: Session = Depends(get_session)
) -> Optional[APIKey]:
    if not x_api_key:
        return None
    
    hashed = hashlib.sha256(x_api_key.encode()).hexdigest()
    stmt = select(APIKey).where(APIKey.hashed_key == hashed, APIKey.is_active == True)
    key_record = session.exec(stmt).first()
    
    if not key_record:
        raise HTTPException(status_code=403, detail="Invalid or inactive API Key.")
        
    # Increment call usage counter
    key_record.total_calls += 1
    session.add(key_record)
    session.commit()
    return key_record

@router.get("", response_model=List[APIKeyResponse])
def get_api_keys(session: Session = Depends(get_session)):
    stmt = select(APIKey).order_by(APIKey.created_at.desc())
    return session.exec(stmt).all()

@router.post("", response_model=APIKeyCreateResponse)
def create_api_key(payload: APIKeyCreate, session: Session = Depends(get_session)):
    # raw_token starts with nlp_live_
    raw_token = f"nlp_live_{secrets.token_hex(24)}"
    hashed = hashlib.sha256(raw_token.encode()).hexdigest()
    # key_prefix: nlp_live_ + first 8 characters of hex token
    prefix = f"nlp_live_{raw_token[9:17]}"
    
    api_key = APIKey(
        name=payload.name,
        key_prefix=prefix,
        hashed_key=hashed,
    )
    session.add(api_key)
    session.commit()
    session.refresh(api_key)
    
    return {
        "id": api_key.id,
        "name": api_key.name,
        "raw_key": raw_token,
        "key_prefix": prefix,
        "created_at": api_key.created_at
    }

@router.delete("/{key_id}")
def revoke_api_key(key_id: int, session: Session = Depends(get_session)):
    api_key = session.get(APIKey, key_id)
    if not api_key:
        raise HTTPException(status_code=404, detail="API Key not found.")
    
    session.delete(api_key)
    session.commit()
    return {"status": "success", "message": f"Successfully deleted API key with ID {key_id}."}
