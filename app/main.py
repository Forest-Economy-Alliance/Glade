from fastapi import FastAPI, BackgroundTasks
from .schemas import RunRequest
from .config import Config

app = FastAPI(title="Image Cleaner Pipeline")

@app.get("/")
def health():
    return {"status": "ok"}
