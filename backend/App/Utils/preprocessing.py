"""
Digital Image Processing (DIP) Preprocessing Pipeline for Brain Tumor Classification.

This module contains the preprocessing steps applied to MRI images before 
feeding them to the DIP model. Steps include:
1. Grayscale conversion (if needed)
2. Gaussian blur for noise reduction  
3. Histogram equalization for contrast enhancement
4. Median filtering for additional noise reduction
5. Edge enhancement using sharpening kernel
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import base64


def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 encoded string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def step1_grayscale(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Step 1: Convert to grayscale if not already.
    
    Args:
        image: Input image (can be color or grayscale)
        
    Returns:
        Tuple of (grayscale_image, step_name)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return gray, "Grayscale Conversion"


def step2_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, str]:
    """
    Step 2: Apply Gaussian blur for noise reduction.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel
        
    Returns:
        Tuple of (blurred_image, step_name)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred, "Gaussian Blur (Noise Reduction)"


def step3_histogram_equalization(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Step 3: Apply histogram equalization for contrast enhancement.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Tuple of (equalized_image, step_name)
    """
    equalized = cv2.equalizeHist(image)
    return equalized, "Histogram Equalization (Contrast Enhancement)"


def step4_median_filter(image: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, str]:
    """
    Step 4: Apply median filter for additional noise reduction.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the median filter kernel
        
    Returns:
        Tuple of (filtered_image, step_name)
    """
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered, "Median Filter (Salt-Pepper Noise Reduction)"


def step5_edge_enhancement(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Step 5: Apply sharpening kernel for edge enhancement.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Tuple of (sharpened_image, step_name)
    """
    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened, "Edge Enhancement (Sharpening)"


def preprocess_image(
    image: np.ndarray, 
    return_steps: bool = False
) -> Dict[str, Any]:
    """
    Apply the full DIP preprocessing pipeline to an image.
    
    Args:
        image: Input image (grayscale or color)
        return_steps: If True, return intermediate step images as base64
        
    Returns:
        Dictionary containing:
        - 'final_image': The fully preprocessed image
        - 'steps': List of preprocessing step info (if return_steps=True)
    """
    steps_info: List[Dict[str, Any]] = []
    
    # Step 1: Grayscale conversion
    current_image, step_name = step1_grayscale(image)
    if return_steps:
        steps_info.append({
            "step_number": 1,
            "step_name": step_name,
            "image_base64": encode_image_to_base64(current_image)
        })
    
    # Step 2: Gaussian blur
    current_image, step_name = step2_gaussian_blur(current_image)
    if return_steps:
        steps_info.append({
            "step_number": 2,
            "step_name": step_name,
            "image_base64": encode_image_to_base64(current_image)
        })
    
    # Step 3: Histogram equalization
    current_image, step_name = step3_histogram_equalization(current_image)
    if return_steps:
        steps_info.append({
            "step_number": 3,
            "step_name": step_name,
            "image_base64": encode_image_to_base64(current_image)
        })
    
    # Step 4: Median filter
    current_image, step_name = step4_median_filter(current_image)
    if return_steps:
        steps_info.append({
            "step_number": 4,
            "step_name": step_name,
            "image_base64": encode_image_to_base64(current_image)
        })
    
    # Step 5: Edge enhancement
    current_image, step_name = step5_edge_enhancement(current_image)
    if return_steps:
        steps_info.append({
            "step_number": 5,
            "step_name": step_name,
            "image_base64": encode_image_to_base64(current_image)
        })
    
    result = {
        "final_image": current_image
    }
    
    if return_steps:
        result["steps"] = steps_info
    
    return result


def preprocess_image_simple(image: np.ndarray) -> np.ndarray:
    """
    Apply the full DIP preprocessing pipeline without returning intermediate steps.
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        The fully preprocessed image
    """
    result = preprocess_image(image, return_steps=False)
    return result["final_image"]
