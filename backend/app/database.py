from sqlmodel import SQLModel, create_engine, Session
from app.core.config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    echo=True,
    connect_args=connect_args,
    pool_pre_ping=True,  # Validate connections before reuse
)

def init_db():
    from app.models.log import LogEntry
    from app.models.model_version import ModelVersion
    from app.models.regex_rule import RegexRule
    from app.models.api_key import APIKey
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine, expire_on_commit=False) as session:
        yield session
