"""CiteClaw web backend — FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="CiteClaw", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}
