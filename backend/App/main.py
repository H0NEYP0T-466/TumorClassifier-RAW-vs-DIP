from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
import joblib
import base64
from pathlib import Path
from typing import Dict, Any, List

from App.Utils.logger import logger
from App.Utils.preprocessing import preprocess_image

# Global variables to store the loaded models
raw_model_data = None
dip_model_data = None

# Model path - models are in the root/models directory
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to repo root
MODELS_DIR = BASE_DIR / "models"
RAW_MODEL_PATH = MODELS_DIR / "raw_svm_model.pkl"
DIP_MODEL_PATH = MODELS_DIR / "dip_svm_model.pkl"

# Class names for prediction output
CLASS_NAMES = ["No Tumor", "Tumor"]


def load_models():
    """Load both RAW and DIP SVM models from disk."""
    global raw_model_data, dip_model_data
    
    raw_loaded = False
    dip_loaded = False
    
    if RAW_MODEL_PATH.exists():
        raw_model_data = joblib.load(RAW_MODEL_PATH)
        logger.info(f"✅ RAW Model Loaded from: {RAW_MODEL_PATH}")
        raw_loaded = True
    else:
        logger.error(f"❌ RAW Model not found at: {RAW_MODEL_PATH}")
    
    if DIP_MODEL_PATH.exists():
        dip_model_data = joblib.load(DIP_MODEL_PATH)
        logger.info(f"✅ DIP Model Loaded from: {DIP_MODEL_PATH}")
        dip_loaded = True
    else:
        logger.warning(f"⚠️ DIP Model not found at: {DIP_MODEL_PATH}")
    
    return raw_loaded, dip_loaded


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 encoded string."""
    # Import from preprocessing module to avoid duplication
    from App.Utils.preprocessing import encode_image_to_base64 as _encode
    return _encode(image)


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
    
    # Load both RAW and DIP SVM models
    logger.info("Loading models...")
    raw_loaded, dip_loaded = load_models()
    
    if raw_loaded:
        logger.info("✅ RAW SVM Model Loaded Successfully!")
    else:
        logger.warning("⚠️ RAW SVM Model not available")
    
    if dip_loaded:
        logger.info("✅ DIP SVM Model Loaded Successfully!")
    else:
        logger.warning("⚠️ DIP SVM Model not available")
    
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
    dip_status = "loaded" if dip_model_data is not None else "not_loaded"
    return {
        "status": "healthy",
        "api": "TumorClassifier",
        "pipelines": {
            "raw": raw_status,
            "dip": dip_status
        }
    }


@app.get("/api/v1/status", tags=["Status"])
async def get_status():
    """
    Get detailed API status and available endpoints.
    """
    logger.info("Status check requested")
    raw_status = "trained" if raw_model_data is not None else "not_trained"
    dip_status = "trained" if dip_model_data is not None else "not_trained"
    return {
        "status": "operational",
        "available_endpoints": {
            "training": {
                "raw_pipeline": "/api/v1/train/raw",
                "dip_pipeline": "/api/v1/train/dip"
            },
            "prediction": {
                "predict": "/api/v1/predict",
                "predict_compare": "/api/v1/predict/compare"
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
            "dip_model": dip_status
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


@app.post("/api/v1/predict/compare", tags=["Prediction"])
async def predict_compare(file: UploadFile = File(...)):
    """
    Compare predictions from both RAW and DIP (preprocessed) models.
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        Comparison results from both models with preprocessing steps visualization
    """
    logger.info("=" * 60)
    logger.info("🔍 Comparison prediction request received")
    
    # Check if at least one model is loaded
    if raw_model_data is None and dip_model_data is None:
        logger.error("❌ Prediction failed: No models loaded")
        raise HTTPException(
            status_code=503, 
            detail="No models loaded. Both RAW and DIP models are unavailable. Please ensure at least one model is trained and loaded."
        )
    
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
        
        # Initialize result structure
        result: Dict[str, Any] = {
            "success": True,
            "original_image_base64": encode_image_to_base64(img),
            "raw_model": None,
            "dip_model": None,
            "preprocessing_steps": []
        }
        
        # ============== RAW MODEL PREDICTION ==============
        if raw_model_data is not None:
            logger.info("🔷 Running RAW model prediction...")
            
            model = raw_model_data["model"]
            pca = raw_model_data["pca"]
            scaler = raw_model_data["scaler"]
            image_size = raw_model_data.get("image_size", (64, 64))
            
            # Resize image to match training size
            img_resized = cv2.resize(img, image_size)
            
            # Flatten and normalize
            img_flat = img_resized.reshape(1, -1).astype(np.float32) / 255.0
            
            # Apply scaler
            img_scaled = scaler.transform(img_flat)
            
            # Apply PCA
            img_pca = pca.transform(img_scaled)
            
            # Make prediction
            prediction = model.predict(img_pca)[0]
            class_name = CLASS_NAMES[prediction]
            
            result["raw_model"] = {
                "prediction": int(prediction),
                "class_name": class_name,
                "message": f"RAW model classified as: {class_name}",
                "input_image_base64": encode_image_to_base64(img_resized)
            }
            
            logger.info(f"🔷 RAW Prediction Result: {class_name}")
        
        # ============== DIP MODEL PREDICTION ==============
        if dip_model_data is not None:
            logger.info("🔶 Running DIP model prediction with preprocessing...")
            
            model = dip_model_data["model"]
            pca = dip_model_data["pca"]
            scaler = dip_model_data["scaler"]
            image_size = dip_model_data.get("image_size", (64, 64))
            
            # Resize image first
            img_resized = cv2.resize(img, image_size)
            
            # Apply preprocessing and get intermediate steps
            preprocess_result = preprocess_image(img_resized, return_steps=True)
            img_preprocessed = preprocess_result["final_image"]
            
            # Store preprocessing steps
            result["preprocessing_steps"] = preprocess_result["steps"]
            
            # Flatten and normalize
            img_flat = img_preprocessed.reshape(1, -1).astype(np.float32) / 255.0
            
            # Apply scaler
            img_scaled = scaler.transform(img_flat)
            
            # Apply PCA
            img_pca = pca.transform(img_scaled)
            
            # Make prediction
            prediction = model.predict(img_pca)[0]
            class_name = CLASS_NAMES[prediction]
            
            result["dip_model"] = {
                "prediction": int(prediction),
                "class_name": class_name,
                "message": f"DIP model classified as: {class_name}",
                "input_image_base64": encode_image_to_base64(img_preprocessed)
            }
            
            logger.info(f"🔶 DIP Prediction Result: {class_name}")
        
        logger.info("✅ Comparison prediction completed")
        logger.info("=" * 60)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comparison prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison prediction failed: {str(e)}")
