from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hybrid Log Classification API"
    
    # Database Config
    DATABASE_URL: str = "sqlite:///./logs.db"
    
    # Redis & Celery Config
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    
    # Groq API Key
    GROQ_API_KEY: str = ""

    # Gemini Config
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"

settings = Settings()
