"""CiteClaw web backend — FastAPI application."""

from fastapi import FastAPI

from api.configs import router as configs_router
from api.papers import router as papers_router
from api.runs import router as runs_router

app = FastAPI(title="CiteClaw", version="0.1.0")

app.include_router(configs_router)
app.include_router(papers_router)
app.include_router(runs_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
