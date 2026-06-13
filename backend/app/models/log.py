from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class LogEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    log_message: str
    target_label: Optional[str] = None
    classification_method: Optional[str] = None
    confidence: Optional[float] = None
    user_corrected: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
