"""
FastAPI backend for Tumor Classifier - Linear SVM Model Training
"""
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tumor Classifier API",
    description="FastAPI backend for training Linear SVM on tumor images",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
IMAGE_SIZE = (64, 64)  # Resize images to a fixed size for consistency


def load_images_from_folder(folder_path: Path, label: int) -> tuple:
    """
    Load images from a folder, convert to grayscale, resize, and flatten.
    
    Args:
        folder_path: Path to the folder containing images
        label: Class label (0 or 1)
    
    Returns:
        Tuple of (features, labels)
    """
    features = []
    labels = []
    
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder_path}")
        return np.array([]), np.array([])
    
    image_files = list(folder_path.glob("*"))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    loaded_count = 0
    for img_path in image_files:
        if img_path.suffix.lower() not in valid_extensions:
            continue
        
        try:
            with Image.open(img_path) as img:
                # Convert to grayscale
                img_gray = img.convert('L')
                # Resize to fixed size
                img_resized = img_gray.resize(IMAGE_SIZE)
                # Flatten to 1D array
                img_array = np.array(img_resized).flatten()
                features.append(img_array)
                labels.append(label)
                loaded_count += 1
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
    
    logger.info(f"Loaded {loaded_count} images from {folder_path}")
    return np.array(features) if features else np.array([]), np.array(labels) if labels else np.array([])


def load_training_data() -> tuple:
    """
    Load all training images from the Dataset folder.
    
    Returns:
        Tuple of (X_train, y_train)
    """
    training_dir = DATASET_DIR / "training"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    logger.info("=" * 50)
    logger.info("Loading training data...")
    logger.info("=" * 50)
    
    # Load tumor images (label = 1)
    tumor_dir = training_dir / "tumer"
    X_tumor, y_tumor = load_images_from_folder(tumor_dir, label=1)
    
    # Load no_tumor images (label = 0)
    no_tumor_dir = training_dir / "no_tumer"
    X_no_tumor, y_no_tumor = load_images_from_folder(no_tumor_dir, label=0)
    
    # Combine datasets
    if len(X_tumor) == 0 and len(X_no_tumor) == 0:
        raise ValueError("No images found in training directories")
    
    if len(X_tumor) > 0 and len(X_no_tumor) > 0:
        X_train = np.vstack([X_tumor, X_no_tumor])
        y_train = np.concatenate([y_tumor, y_no_tumor])
    elif len(X_tumor) > 0:
        X_train = X_tumor
        y_train = y_tumor
    else:
        X_train = X_no_tumor
        y_train = y_no_tumor
    
    # Log class distribution
    logger.info("=" * 50)
    logger.info("Dataset Statistics:")
    logger.info("=" * 50)
    logger.info(f"Total training images: {len(y_train)}")
    logger.info(f"  - Tumor images: {np.sum(y_train == 1)}")
    logger.info(f"  - No tumor images: {np.sum(y_train == 0)}")
    logger.info(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
    logger.info(f"Feature vector size: {X_train.shape[1]}")
    
    return X_train, y_train


def train_linear_svm(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train a Linear SVM model on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Tuple of (trained model, scaler)
    """
    logger.info("=" * 50)
    logger.info("Training Linear SVM Model...")
    logger.info("=" * 50)
    
    # Normalize features using StandardScaler
    logger.info("Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Linear SVM
    logger.info("Training LinearSVC...")
    model = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    logger.info("Training completed successfully!")
    
    return model, scaler


def save_model(model, scaler) -> None:
    """
    Save the trained model and scaler to disk.
    """
    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to: {MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Tumor Classifier API is running",
        "version": "1.0.0"
    }


@app.post("/train")
async def train_model():
    """
    Trigger training of the Linear SVM model.
    
    This endpoint:
    1. Loads training images from the Dataset folder
    2. Converts images to grayscale and flattens them
    3. Trains a Linear SVM model
    4. Saves the model as model.pkl in the model folder
    """
    try:
        logger.info("=" * 50)
        logger.info("Starting model training...")
        logger.info("=" * 50)
        
        # Load training data
        X_train, y_train = load_training_data()
        
        # Train the model
        model, scaler = train_linear_svm(X_train, y_train)
        
        # Save the model
        save_model(model, scaler)
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Model trained and saved successfully",
            "details": {
                "total_images": int(len(y_train)),
                "tumor_images": int(np.sum(y_train == 1)),
                "no_tumor_images": int(np.sum(y_train == 0)),
                "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
                "feature_size": int(X_train.shape[1]),
                "model_path": str(MODEL_PATH)
            }
        }
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info("=" * 50)
        
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/model/status")
async def model_status():
    """Check if a trained model exists"""
    model_exists = MODEL_PATH.exists()
    scaler_exists = SCALER_PATH.exists()
    
    return {
        "model_exists": model_exists,
        "scaler_exists": scaler_exists,
        "model_path": str(MODEL_PATH) if model_exists else None,
        "scaler_path": str(SCALER_PATH) if scaler_exists else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
