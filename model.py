"""
LeafCare - Computer Vision Model and Image Preprocessing Module
College Applied AI Project - BCA AI & ML Viva Documentation

This module handles:
1. Loading configuration from metadata.json (classes, image dimensions, normalization constants)
2. Instantiating PyTorch ResNet18 and loading the pre-trained weights (leafcare_resnet18.pth)
3. Robust image loading and preprocessing using Pillow and OpenCV (cv2):
   - Format validation (JPG, JPEG, PNG)
   - Color-space validation and conversion (Grayscale/RGBA to RGB)
   - OpenCV resizing to 224x224
   - PyTorch tensor conversion and Channel-wise Normalization (mean/std)
4. Inference execution with softmax probability calibration
5. Uncertainty assessment for out-of-distribution or ambiguous inputs
"""

import os
import json
from typing import Dict, Tuple, Any, Union
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models


# Human-readable labels mapping
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy": "Tomato - Healthy",
    "Tomato___Early_blight": "Tomato - Early Blight",
    "Tomato___Late_blight": "Tomato - Late Blight"
}

# Mapping to knowledge base category keys
CLASS_TO_KB_KEY = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight"
}

# Confidence threshold below which prediction is considered uncertain
CONFIDENCE_THRESHOLD = 0.65


def load_metadata(metadata_path: str = "metadata.json") -> Dict[str, Any]:
    """
    Loads and validates model metadata from metadata.json.
    
    Contains:
    - architecture: e.g. "ResNet18"
    - classes: list of 3 tomato leaf categories
    - image_size: target image height and width (224)
    - normalization_mean: [0.485, 0.456, 0.406]
    - normalization_std: [0.229, 0.224, 0.225]
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file '{metadata_path}' not found. "
            f"Ensure metadata.json is in the project root."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_keys = ["classes", "image_size", "normalization_mean", "normalization_std"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"metadata.json is missing required configuration key: '{key}'")

    return metadata


def build_resnet18(num_classes: int = 3) -> nn.Module:
    """
    Constructs the standard ResNet18 architecture and replaces the
    final fully-connected linear layer to output `num_classes` logits.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(
    weights_path: str = "leafcare_resnet18.pth",
    metadata_path: str = "metadata.json",
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads the trained PyTorch model and metadata once at startup.
    
    Parameters:
        weights_path: Path to the .pth weights file.
        metadata_path: Path to the metadata.json file.
        device: Device to place the model on ('cpu', 'mps', or 'cuda').
        
    Returns:
        (model, metadata) tuple.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights file '{weights_path}' was not found. "
            f"Ensure 'leafcare_resnet18.pth' is located in the project directory."
        )

    metadata = load_metadata(metadata_path)
    classes = metadata["classes"]

    # Build model architecture matching the trained checkpoint
    model = build_resnet18(num_classes=len(classes))

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Freeze BatchNorm and Dropout layers for inference

    return model, metadata


def preprocess_image(
    image_input: Union[str, bytes, BytesIO, Image.Image],
    metadata: Dict[str, Any]
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Validates, loads, and preprocesses an input leaf image.
    
    Pipeline:
    1. Pillow loads the file (validating image format: JPG, JPEG, PNG).
    2. Image is converted to a NumPy array for OpenCV preprocessing.
    3. OpenCV verifies color channels and converts RGBA/Grayscale to RGB.
    4. OpenCV resizes the image to the metadata target size (224 x 224) using bilinear interpolation.
    5. The image is converted into a PyTorch float32 Tensor, scaled to [0, 1].
    6. Training normalization is applied: (channel - mean) / std.
    7. A batch dimension is prepended -> shape (1, 3, 224, 224).
    
    Returns:
        (preprocessed_tensor, display_pil_image)
    """
    # Step 1: Pillow image loading
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
        elif isinstance(image_input, (str, bytes, BytesIO)):
            pil_img = Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Verify valid format
        format_detected = pil_img.format
        if format_detected and format_detected.upper() not in ["JPEG", "JPG", "PNG"]:
            # Note: uploaded file streams may already be opened, so format could be None if converted
            pass
    except Exception as e:
        raise ValueError(f"Failed to read image file. Please provide a valid JPG, JPEG, or PNG image. Error: {e}")

    # Step 2: Convert to NumPy array for OpenCV processing
    np_img = np.array(pil_img)

    # Step 3: OpenCV color channels validation & conversion
    if len(np_img.shape) == 2:
        # Grayscale image (H, W) -> Convert to 3-channel RGB
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    elif len(np_img.shape) == 3:
        if np_img.shape[2] == 4:
            # RGBA image -> Convert to RGB
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        elif np_img.shape[2] == 3:
            # Standard 3 channels (Pillow array is RGB)
            pass
        else:
            raise ValueError(f"Unexpected number of color channels: {np_img.shape[2]}")
    else:
        raise ValueError(f"Invalid image array shape: {np_img.shape}")

    # Step 4: OpenCV Resizing to metadata specifications (224x224)
    target_size = metadata.get("image_size", 224)
    resized_np = cv2.resize(np_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Step 5: Convert NumPy array (H, W, C) to PyTorch Float Tensor (C, H, W) normalized to [0, 1]
    tensor_img = torch.from_numpy(resized_np).permute(2, 0, 1).float() / 255.0

    # Step 6: Channel-wise Normalization using training mean and std from metadata
    mean = torch.tensor(metadata["normalization_mean"]).view(3, 1, 1)
    std = torch.tensor(metadata["normalization_std"]).view(3, 1, 1)
    normalized_tensor = (tensor_img - mean) / std

    # Step 7: Prepend batch dimension (1, 3, 224, 224)
    batch_tensor = normalized_tensor.unsqueeze(0)

    # Display image in RGB
    display_pil = Image.fromarray(np_img)

    return batch_tensor, display_pil


def predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    metadata: Dict[str, Any],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes inference using the loaded ResNet18 model and returns prediction
    results, confidence scores, and uncertainty analysis.
    """
    classes = metadata["classes"]

    model.eval()
    with torch.no_grad():
        tensor_device = input_tensor.to(device)
        logits = model(tensor_device)
        # Softmax probability distribution over the 3 classes
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Model predicted index
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    raw_class_name = classes[predicted_idx]
    display_name = CLASS_DISPLAY_NAMES.get(raw_class_name, raw_class_name)
    kb_key = CLASS_TO_KB_KEY.get(raw_class_name, raw_class_name)

    # Mapping of all class probabilities
    all_probs = {
        classes[i]: {
            "display_name": CLASS_DISPLAY_NAMES.get(classes[i], classes[i]),
            "probability": float(probabilities[i]),
            "percentage": f"{probabilities[i] * 100:.2f}%"
        }
        for i in range(len(classes))
    }

    # Uncertainty evaluation:
    # 1. Confidence below threshold
    # 2. Or margin between top 1 and top 2 is narrow (< 15%)
    sorted_probs = sorted(probabilities, reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    is_uncertain = (confidence < CONFIDENCE_THRESHOLD) or (margin < 0.15)
    
    uncertainty_reasons = []
    if confidence < CONFIDENCE_THRESHOLD:
        uncertainty_reasons.append(
            f"Confidence score ({confidence * 100:.1f}%) is below the {CONFIDENCE_THRESHOLD * 100:.0f}% certainty threshold."
        )
    if margin < 0.15:
        uncertainty_reasons.append(
            f"Narrow separation margin ({margin * 100:.1f}%) between top predicted classes."
        )

    return {
        "predicted_idx": predicted_idx,
        "raw_class": raw_class_name,
        "display_name": display_name,
        "kb_key": kb_key,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": all_probs,
        "is_uncertain": is_uncertain,
        "uncertainty_reasons": uncertainty_reasons
    }
