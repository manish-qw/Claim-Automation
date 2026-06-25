import os
# Force PaddleOCR to not explode CPU/RAM thread counts and avoid OneDNN C++ crashes
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routers.claims import router as claims_router
try:
    from api.routers.fraud_bridge import router as fraud_bridge_router
except Exception as exc:
    fraud_bridge_router = None
    print(f"[FRAUD BRIDGE] disabled at startup: {exc}")

app = FastAPI(
    title="CLAIMOS AI — Claim Automation API",
    version="1.0.0",
    docs_url="/docs"
)

# CORS configuration to allow local frontend access (supporting port 3000, 5173, and local subnets)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(claims_router, prefix="/v1/claims", tags=["Claims"])
if fraud_bridge_router is not None:
    app.include_router(fraud_bridge_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "claimos-api"}

@app.post("/api/usage-metrics/events")
async def usage_metrics_events(_: Request):
    # Frontend telemetry sink; intentionally lightweight and non-blocking.
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
