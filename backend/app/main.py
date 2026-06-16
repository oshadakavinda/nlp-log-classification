from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import logs, keys
from app.database import init_db
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Hybrid Log Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(keys.router, prefix="/api/keys", tags=["API Keys"])

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Log Classification API is running"}
