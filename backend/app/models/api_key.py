from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class APIKey(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    key_prefix: str
    hashed_key: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    total_calls: int = Field(default=0)
