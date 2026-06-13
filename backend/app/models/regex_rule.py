from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class RegexRule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pattern: str
    target_label: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
