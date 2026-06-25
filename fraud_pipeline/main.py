"""
CLAIMOS AI — Fraud & Investigation Intelligence Layer
Main FastAPI application entry point.
"""
import os
from dotenv import load_dotenv
# Explicitly load .env from the fraud_pipeline directory BEFORE anything else
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'), override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router

app = FastAPI(
    title="CLAIMOS AI — Fraud Intelligence Pipeline",
    description=(
        "Trust-Aware Fraud & Investigation Intelligence Layer for "
        "autonomous life insurance death claims adjudication."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "CLAIMOS Fraud Intelligence Pipeline",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/fraud/health",
        "analyze": "POST /fraud/analyze",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
