from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
import joblib
from pathlib import Path

from App.Utils.logger import logger

# Global variable to store the loaded model
raw_model_data = None

# Model path - models are in the root/models directory
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to repo root
MODELS_DIR = BASE_DIR / "models"
RAW_MODEL_PATH = MODELS_DIR / "raw_svm_model.pkl"

# Class names for prediction output
CLASS_NAMES = ["No Tumor", "Tumor"]


def load_model():
    """Load the RAW SVM model from disk."""
    global raw_model_data
    
    if RAW_MODEL_PATH.exists():
        raw_model_data = joblib.load(RAW_MODEL_PATH)
        logger.info(f"✅ Model Loaded from: {RAW_MODEL_PATH}")
        return True
    else:
        logger.error(f"❌ Model not found at: {RAW_MODEL_PATH}")
        return False


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
    
    # Load RAW SVM model
    logger.info("Loading RAW pipeline model...")
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("✅ RAW SVM Model Loaded Successfully!")
    else:
        logger.warning("⚠️ RAW SVM Model not available - prediction endpoint will not work")
    
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
    raw_status = "loaded" if raw_model_data is not None else "not_loaded"
    return {
        "status": "healthy",
        "api": "TumorClassifier",
        "pipelines": {
            "raw": raw_status,
            "dip": "not_loaded"
        }
    }


@app.get("/api/v1/status", tags=["Status"])
async def get_status():
    """
    Get detailed API status and available endpoints.
    """
    logger.info("Status check requested")
    raw_status = "trained" if raw_model_data is not None else "not_trained"
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
            "raw_model": raw_status,
            "dip_model": "not_trained"
        }
    }


@app.post("/api/v1/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded MRI image contains a tumor.
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        Prediction result with class name and confidence
    """
    logger.info("=" * 40)
    logger.info("🔍 Prediction request received")
    
    # Check if model is loaded
    if raw_model_data is None:
        logger.error("❌ Prediction failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model is trained and loaded.")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        logger.info(f"📁 Received file: {file.filename}, Size: {len(contents)} bytes")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image in grayscale
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logger.error("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image file. Could not decode the image.")
        
        logger.info(f"📐 Original image shape: {img.shape}")
        
        # Get model components
        model = raw_model_data["model"]
        pca = raw_model_data["pca"]
        scaler = raw_model_data["scaler"]
        image_size = raw_model_data.get("image_size", (64, 64))
        
        logger.info(f"🔧 Preprocessing image to size: {image_size}")
        
        # Resize image to match training size
        img_resized = cv2.resize(img, image_size)
        
        # Flatten and normalize
        img_flat = img_resized.reshape(1, -1).astype(np.float32) / 255.0
        
        # Apply scaler
        img_scaled = scaler.transform(img_flat)
        
        # Apply PCA
        img_pca = pca.transform(img_scaled)
        
        logger.info(f"📊 Feature shape after PCA: {img_pca.shape}")
        
        # Make prediction
        prediction = model.predict(img_pca)[0]
        class_name = CLASS_NAMES[prediction]
        
        logger.info(f"🎯 Prediction Result: {class_name} (class={prediction})")
        logger.info("=" * 40)
        
        return {
            "success": True,
            "prediction": int(prediction),
            "class_name": class_name,
            "message": f"The scan has been classified as: {class_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
