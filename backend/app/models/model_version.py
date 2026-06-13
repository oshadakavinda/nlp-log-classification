from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    version_tag: str  # e.g., "v_20260613_172000"
    dataset_name: str
    num_records: int
    accuracy: float
    metrics_json: str  # JSON-encoded classification report
    is_active: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
