# src/main.py

from fastapi import FastAPI
from src.api import endpoints

app = FastAPI(
    title="Audio Transcription Service",
    description="A microservice to separate vocals and transcribe audio.",
    version="1.0.0"
)

app.include_router(endpoints.router)

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok"}