"""
DIP (Digital Image Processing) Model Training Pipeline.

This script trains a Linear SVM model on preprocessed (DIP) images.
Uses the same model architecture as the RAW model:
- 64x64 image size
- StandardScaler normalization
- PCA with 256 components
- LinearSVC classifier
"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from App.Utils.logger import logger
from App.Utils.preprocessing import preprocess_image_simple


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed path to your dataset
DATASET_DIR = Path(r"X:\file\FAST_API\TumorClassifier-RAW-vs-DIP\backend\Dataset")
TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"

# Models directory (inside backend)
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model and metrics save paths
DIP_MODEL_PATH = MODELS_DIR / "dip_svm_model.pkl"
DIP_METRICS_PATH = MODELS_DIR / "dip_metrics.pkl"

IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42
CLASS_NAMES = ["No Tumor", "Tumor"]


# ============================================================
# DATA LOADING (WITH DIP PREPROCESSING)
# ============================================================

def load_images_from_folder_with_preprocessing(folder_path: Path, label: int) -> Tuple[list, list]:
    """Load all images from a folder, apply DIP preprocessing, and assign a label."""
    images = []
    labels = []
    
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder_path}")
        return images, labels
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_file in folder_path.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            try:
                # Read image in grayscale
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize first
                    img = cv2.resize(img, IMAGE_SIZE)
                    
                    # Apply DIP preprocessing
                    img_preprocessed = preprocess_image_simple(img)
                    
                    images.append(img_preprocessed)
                    labels.append(label)
                    
            except Exception as e:
                logger.error(f"Error loading image {img_file}: {e}")
    
    logger.info(f"  Loaded {len(images)} preprocessed images from {folder_path.name}/ (label={label})")
    return images, labels


def load_dataset_with_preprocessing(dataset_type: str = "training") -> Tuple[np.ndarray, np.ndarray]:
    """Load the complete dataset with DIP preprocessing (training or testing)."""
    
    logger.info(f"Loading {dataset_type.upper()} dataset with DIP preprocessing...")
    
    base_dir = TRAINING_DIR if dataset_type == "training" else TESTING_DIR
    
    logger.info(f"  Looking in: {base_dir}")
    
    all_images = []
    all_labels = []
    
    # Load tumor images (label = 1)
    tumor_path = base_dir / "tumor"
    tumor_images, tumor_labels = load_images_from_folder_with_preprocessing(tumor_path, 1)
    all_images.extend(tumor_images)
    all_labels.extend(tumor_labels)
    
    # Load no_tumor images (label = 0)
    no_tumor_path = base_dir / "no_tumor"
    no_tumor_images, no_tumor_labels = load_images_from_folder_with_preprocessing(no_tumor_path, 0)
    all_images.extend(no_tumor_images)
    all_labels.extend(no_tumor_labels)
    
    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    
    logger.info(f"  Total: {len(X)} images | Tumor: {sum(y == 1)} | No Tumor: {sum(y == 0)}")
    
    return X, y


# ============================================================
# METRICS CALCULATION
# ============================================================

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Calculate all evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": classification_report(
            y_true, y_pred, 
            target_names=CLASS_NAMES, 
            output_dict=True
        )
    }
    
    # AUC-ROC if probabilities available
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        metrics["auc_roc"] = round(float(auc), 4)
        
        # ROC Curve data
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        metrics["roc_curve"] = {
            "fpr": [round(x, 4) for x in fpr.tolist()],
            "tpr": [round(x, 4) for x in tpr.tolist()]
        }
    
    return metrics


# ============================================================
# MODEL TRAINING
# ============================================================

def train_dip_model() -> Dict[str, Any]:
    """Train the DIP (preprocessed) Linear SVM model."""
    
    logger.info("=" * 60)
    logger.info("🧠 STARTING DIP PIPELINE TRAINING (LINEAR SVM + PCA + PREPROCESSING)")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # -------------------- Load Data --------------------
    logger.info("\n📂 STEP 1: Loading Dataset with DIP Preprocessing")
    logger.info("-" * 40)
    
    X_train, y_train = load_dataset_with_preprocessing("training")
    X_test, y_test = load_dataset_with_preprocessing("testing")
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Training or Testing dataset is empty!")
        return {"success": False, "message": "Dataset not found"}
    
    logger.info(f"\n  Training set shape: {X_train.shape}")
    logger.info(f"  Testing set shape: {X_test.shape}")
    
    # -------------------- Flatten & Scale --------------------
    logger.info("\n🔧 STEP 2: Flattening and Scaling Features")
    logger.info("-" * 40)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    
    # Standardize features
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # -------------------- PCA --------------------
    logger.info("\n🌀 STEP 3: Applying PCA (256 components)")
    logger.info("-" * 40)
    
    pca = PCA(n_components=256, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    logger.info(f"  Training features shape after PCA: {X_train_pca.shape}")
    logger.info(f"  Testing features shape after PCA: {X_test_pca.shape}")
    
    # -------------------- Train LinearSVC --------------------
    logger.info("\n🚀 STEP 4: Training LinearSVC Model")
    logger.info("-" * 40)
    
    model = LinearSVC(
        random_state=RANDOM_STATE,
        verbose=1,
        max_iter=5000
    )
    
    model.fit(X_train_pca, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"  ✅ Training completed in {training_time:.2f} seconds")
    
    # -------------------- Evaluate Model --------------------
    logger.info("\n📊 STEP 5: Evaluating Model")
    y_pred = model.predict(X_test_pca)
    
    # LinearSVC does not support predict_proba by default
    y_prob = None  # Skip AUC-ROC if not needed
    
    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    
    # Log metrics
    logger.info(f"\n  📈 RESULTS ON TEST SET:")
    logger.info(f"  {'─'*30}")
    logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"  Recall: {metrics['recall']*100:.2f}%")
    logger.info(f"  F1 Score: {metrics['f1_score']*100:.2f}%")
    
    # -------------------- Save Model & Metrics --------------------
    model_data = {
        "model": model,
        "pca": pca,
        "scaler": scaler,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "preprocessing": "dip"  # Flag to indicate this model requires DIP preprocessing
    }
    joblib.dump(model_data, DIP_MODEL_PATH)
    metrics["training_time_seconds"] = round(training_time, 2)
    joblib.dump(metrics, DIP_METRICS_PATH)
    
    logger.info(f"  ✅ Model saved to: {DIP_MODEL_PATH}")
    logger.info(f"  ✅ Metrics saved to: {DIP_METRICS_PATH}")
    
    return {"success": True, "metrics": metrics, "model_path": str(DIP_MODEL_PATH)}


def load_dip_model():
    """Load the saved DIP model for predictions."""
    if DIP_MODEL_PATH.exists():
        model_data = joblib.load(DIP_MODEL_PATH)
        logger.info(f"✅ DIP Model loaded from: {DIP_MODEL_PATH}")
        return model_data
    else:
        logger.error(f"DIP Model not found at: {DIP_MODEL_PATH}")
        return None


def get_dip_model_metrics() -> Dict[str, Any]:
    """Load and return saved metrics for DIP pipeline."""
    if DIP_METRICS_PATH.exists():
        metrics = joblib.load(DIP_METRICS_PATH)
        return {"success": True, "metrics": metrics}
    else:
        return {"success": False, "message": "No metrics found for DIP pipeline"}


def is_dip_model_trained() -> bool:
    """Check if DIP model has been trained."""
    return DIP_MODEL_PATH.exists()


if __name__ == "__main__":
    logger.info("💻 Starting DIP model training...")
    result = train_dip_model()
    logger.info(f"💖 Training Result: {result}")
