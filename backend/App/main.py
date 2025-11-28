from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from App.Utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting TumorClassifier API Server...")
    logger.info("=" * 60)
    logger.info("Initializing application components...")
    
    # TODO: Load models here when they exist
    # logger.info("Loading RAW pipeline model...")
    # logger.info("Loading DIP pipeline model...")
    
    logger.info("✅ Application startup complete!")
    logger.info("-" * 60)
    
    yield  # Server is running
    
    # Shutdown
    logger.info("-" * 60)
    logger.info("🛑 Shutting down TumorClassifier API Server...")
    logger.info("Cleaning up resources...")
    logger.info("✅ Shutdown complete.  Goodbye!")
    logger.info("=" * 60)


# Initialize FastAPI app
app = FastAPI(
    title="TumorClassifier API",
    description="Brain Tumor Detection API - RAW vs DIP Pipeline Comparison",
    version="1.0. 0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Health check.
    """
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "message": "TumorClassifier API is running! ",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    logger.debug("Health check performed")
    return {
        "status": "healthy",
        "api": "TumorClassifier",
        "pipelines": {
            "raw": "not_loaded",  # Will update when models are trained
            "dip": "not_loaded"
        }
    }


@app.get("/api/v1/status", tags=["Status"])
async def get_status():
    """
    Get detailed API status and available endpoints.
    """
    logger.info("Status check requested")
    return {
        "status": "operational",
        "available_endpoints": {
            "training": {
                "raw_pipeline": "/api/v1/train/raw",
                "dip_pipeline": "/api/v1/train/dip"
            },
            "prediction": {
                "predict": "/api/v1/predict"
            },
            "metrics": {
                "get_metrics": "/api/v1/metrics",
                "compare": "/api/v1/metrics/compare"
            },
            "logs": {
                "get_logs": "/api/v1/logs"
            }
        },
        "models": {
            "raw_model": "not_trained",
            "dip_model": "not_trained"
        }
    }
