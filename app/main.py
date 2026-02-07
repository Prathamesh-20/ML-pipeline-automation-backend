"""
ML Pipeline Automation - FastAPI Backend
A commercial-grade ML pipeline automation platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routers import upload, analysis, preprocessing, training, predict

# Create FastAPI app
app = FastAPI(
    title="ML Pipeline Automation API",
    description="A user-centric machine learning pipeline automation platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(preprocessing.router, prefix="/api", tags=["Preprocessing"])
app.include_router(training.router, prefix="/api", tags=["Training"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "ML Pipeline Automation API is running!",
        "docs": "/api/docs"
    }


@app.get("/api/health")
async def health_check():
    """API health check"""
    return {"status": "ok", "version": "1.0.0"}
