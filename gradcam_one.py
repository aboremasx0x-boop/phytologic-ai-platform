# =========================
# gradcam_one.py  (انسخه كما هو)
# =========================
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from PIL import Image


def _find_last_conv_layer(model: torch.nn.Module):
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


def _to_bgr_uint8(pil_img: Image.Image, size=(224, 224)):
    img = pil_img.resize(size)
    arr = np.array(img)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def make_gradcam_overlay_b64(model: torch.nn.Module, x: torch.Tensor, pil_img: Image.Image) -> str:
    """
    يرجع صورة overlay بصيغة base64 (PNG)
    """
    model.eval()

    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")

    features = []
    grads = []

    def fwd_hook(module, inp, out):
        features.append(out)

    def bwd_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # forward
    logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())

    # backward on predicted class
    model.zero_grad(set_to_none=True)
    score = logits[0, pred]
    score.backward()

    # cleanup hooks
    h1.remove()
    h2.remove()

    fmap = features[0]           # [1, C, H, W]
    grad = grads[0]              # [1, C, H, W]

    # weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = torch.sum(weights * fmap, dim=1).squeeze(0)     # [H, W]
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()

    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    base = _to_bgr_uint8(pil_img, size=(224, 224))

    overlay = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0.0)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise ValueError("Failed to encode Grad-CAM overlay.")

    return base64.b64encode(buf.tobytes()).decode("utf-8")
