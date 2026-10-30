"""CropCare ΓÇö Grad-CAM Visual Explainability Module.

This module provides Gradient-weighted Class Activation Mapping (Grad-CAM)
for the trained MobileNetV3-Small leaf classifier. It visualizes which spatial
regions (lesions, discolorations, margins) of the leaf photograph influenced
the neural network's decision, turning a 'black-box' model into an interpretable
agronomic diagnostic tool.

Scientific Principles:
1. Grad-CAM computes gradients of the target class score with respect to feature maps
   in the final convolutional layer (model.features[-1]).
2. Positive gradients highlight regions where increased intensity increases model confidence.
3. Overlays highlight authentic pathology lesions, allowing agronomists to verify that
   predictions are not driven by background paper, shadows, or camera artifacts.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _apply_colormap(cam_norm: np.ndarray) -> np.ndarray:
    """Apply an agricultural heat colormap (Blue/Cyan -> Yellow -> Red).

    Args:
        cam_norm: 2D float array in range [0, 1].

    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    # Simple, high-quality pure-NumPy Jet/Turbo colormap without requiring matplotlib
    # Values: 0.0 -> Blue (0, 0, 180), 0.35 -> Cyan (0, 200, 255), 0.65 -> Yellow (255, 220, 0), 1.0 -> Red (220, 20, 0)
    h, w = cam_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Vectorized piecewise color ramps
    x = np.clip(cam_norm, 0.0, 1.0)
    
    # Red channel
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    # Green channel
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    # Blue channel
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    rgb[:, :, 0] = (r * 255).astype(np.uint8)
    rgb[:, :, 1] = (g * 255).astype(np.uint8)
    rgb[:, :, 2] = (b * 255).astype(np.uint8)
    return rgb


def generate_gradcam(
    image: Union[str, bytes, Image.Image, np.ndarray],
    target_class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """Generate Grad-CAM activation heatmap overlay for a leaf image.

    Args:
        image: Leaf input image.
        target_class_idx: Target class index to explain. If None, uses top predicted class.
        alpha: Blend ratio for heatmap overlay (0.0 = original only, 1.0 = heatmap only).

    Returns:
        Dictionary containing:
            - 'overlay_image': PIL Image of original with blended heatmap.
            - 'heatmap_image': PIL Image of raw colormapped heatmap.
            - 'peak_intensity': Float peak activation value (0 to 1).
            - 'focus_description': Human-readable summary of attention concentration.
            - 'success': Boolean indicating whether Grad-CAM succeeded.
    """
    from src.cropcare.image_engine import get_model, preprocess_image, predict_image

    # Standardize input image to PIL
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    orig_w, orig_h = pil_img.size

    if not HAS_TORCH:
        return _fallback_contrast_saliency(pil_img, alpha)

    model, mapping = get_model()
    if model is None:
        return _fallback_contrast_saliency(pil_img, alpha)

    try:
        model.eval()

        # Hooks to capture activations and gradients
        feature_activations: list[torch.Tensor] = []
        feature_gradients: list[torch.Tensor] = []

        # Target layer: last conv block in MobileNetV3 features
        target_layer = model.features[-1]

        def forward_hook(module, inp, out):
            feature_activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            feature_gradients.append(grad_out[0])

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        # Preprocess image
        input_tensor = preprocess_image(pil_img)
        input_tensor.requires_grad = True

        # Forward pass
        logits = model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]

        # Backward pass
        model.zero_grad()
        score.backward(retain_graph=True)

        # Clean up hooks immediately
        h_fwd.remove()
        h_bwd.remove()

        if not feature_activations or not feature_gradients:
            return _fallback_contrast_saliency(pil_img, alpha)

        # Grad-CAM calculation
        # Activations shape: (1, Channels, H, W)
        # Gradients shape: (1, Channels, H, W)
        act = feature_activations[0].detach()
        grad = feature_gradients[0].detach()

        # Global average pool gradients along spatial dimensions to get channel weights
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * act, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive influence (ReLU)

        cam_np = cam.squeeze().cpu().numpy()
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)

        if max_val > min_val:
            cam_norm = (cam_np - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Resize CAM to match original image dimensions using PIL bilinear
        cam_pil = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0

        # Apply agricultural colormap
        cam_colored = _apply_colormap(cam_resized)
        raw_heatmap_pil = Image.fromarray(cam_colored)

        # Blend with original photograph
        orig_np = np.array(pil_img)
        blended = (orig_np * (1.0 - alpha) + cam_colored * alpha).clip(0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended)

        # Qualitative attention focus summary
        high_attention_ratio = float(np.mean(cam_resized > 0.60))
        peak_intensity = float(np.max(cam_resized))

        if high_attention_ratio < 0.25:
            focus_desc = "Neural attention tightly focused on localized necrotic lesion clusters on the leaf blade."
        elif high_attention_ratio < 0.55:
            focus_desc = "Broad attention across multi-focal leaf spots and interveinal chlorotic tissue."
        else:
            focus_desc = "Diffuse spatial attention across leaf surface, characteristic of systemic chlorosis or mosaic patterning."

        return {
            "overlay_image": overlay_pil,
            "heatmap_image": raw_heatmap_pil,
            "peak_intensity": peak_intensity,
            "focus_description": focus_desc,
            "success": True,
        }

    except Exception as e:
        # Graceful fallback without crashing
        return _fallback_contrast_saliency(pil_img, alpha, error_msg=str(e))


def _fallback_contrast_saliency(
    pil_img: Image.Image,
    alpha: float = 0.45,
    error_msg: Optional[str] = None,
) -> Dict[str, Any]:
    """Resilient fallback creating high-frequency edge/lesion saliency map."""
    orig_w, orig_h = pil_img.size
    gray = np.array(pil_img.convert("L"), dtype=np.float32)

    # Compute local contrast difference (lesion highlight proxy)
    from PIL import ImageFilter
    blurred = np.array(pil_img.convert("L").filter(ImageFilter.GaussianBlur(radius=5)), dtype=np.float32)
    diff = np.abs(gray - blurred)
    max_d = np.max(diff)
    diff_norm = diff / (max_d if max_d > 0 else 1.0)

    cam_colored = _apply_colormap(diff_norm)
    raw_heatmap_pil = Image.fromarray(cam_colored)

    orig_np = np.array(pil_img)
    blended = (orig_np * (1.0 - alpha) + cam_colored * alpha).clip(0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(blended)

    return {
        "overlay_image": overlay_pil,
        "heatmap_image": raw_heatmap_pil,
        "peak_intensity": float(np.max(diff_norm)),
        "focus_description": "Edge-contrast lesion saliency map generated via optical frequency analysis.",
        "success": False,
        "warning": error_msg or "Model gradient hooks unavailable; using optical lesion saliency fallback.",
    }
