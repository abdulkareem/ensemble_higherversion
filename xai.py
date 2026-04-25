from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import ensure_binary_output


def _find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layers found for Grad-CAM.")
    return conv_layers[-1]


def grad_cam_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """Generate Grad-CAM for binary segmentation logits."""
    model.eval()
    layer = target_layer or _find_last_conv(model)

    acts: Dict[str, torch.Tensor] = {}
    grads: Dict[str, torch.Tensor] = {}

    def _save_acts(_, __, output):
        acts["value"] = output

    def _save_grads(_, grad_input, grad_output):
        del grad_input
        grads["value"] = grad_output[0]

    h1 = layer.register_forward_hook(_save_acts)
    h2 = layer.register_full_backward_hook(_save_grads)

    logits = ensure_binary_output(model(image))
    score = torch.sigmoid(logits).mean()

    model.zero_grad(set_to_none=True)
    score.backward()

    h1.remove()
    h2.remove()

    a = acts["value"]
    g = grads["value"]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = (weights * a).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def mamba_state_response(model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    """Proxy visualization of Mamba state response via last feature activations."""
    activations: Dict[str, torch.Tensor] = {}

    target_module = None
    if hasattr(model, "mamba") and len(getattr(model, "mamba")) > 0:
        target_module = model.mamba[-1]
    else:
        target_module = _find_last_conv(model)

    def _save(_, __, output):
        activations["value"] = output

    hook = target_module.register_forward_hook(_save)
    with torch.no_grad():
        _ = model(image)
    hook.remove()

    fmap = activations["value"]
    if fmap.ndim == 4:
        fmap = fmap.abs().mean(dim=1, keepdim=True)
    fmap = F.interpolate(fmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    fmap = fmap[0, 0].detach().cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    return fmap


def cross_paradigm_xai(
    mamba_model: torch.nn.Module,
    transformer_model: torch.nn.Module,
    cnn_model: torch.nn.Module,
    image: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Produce branchwise XAI maps + consensus/disagreement for manuscript figures."""
    mamba_cam = grad_cam_heatmap(mamba_model, image)
    transformer_cam = grad_cam_heatmap(transformer_model, image)
    cnn_cam = grad_cam_heatmap(cnn_model, image)
    mamba_ssm = mamba_state_response(mamba_model, image)

    stacked = np.stack([mamba_cam, transformer_cam, cnn_cam], axis=0)
    consensus = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0)

    return {
        "mamba_gradcam": mamba_cam,
        "mamba_state_response": mamba_ssm,
        "transformer_attention_proxy": transformer_cam,
        "cnn_gradcam": cnn_cam,
        "consensus_map": consensus,
        "disagreement_map": disagreement,
    }


def overlay_heatmap(image_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay CAM on the RGB input image for qualitative explainability figures."""
    image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = np.clip(image * std + mean, 0, 1)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    blend = np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)
    return (blend * 255).astype(np.uint8)
