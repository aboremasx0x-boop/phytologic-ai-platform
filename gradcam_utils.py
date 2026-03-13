# gradcam_utils.py
# (نسخة مُدقّقة ومحسّنة بدون حذف الفكرة — فقط إصلاحات تمنع الأعطال + إزالة اعتماد matplotlib)

import base64
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    ResNet: الأفضل هو آخر Conv داخل layer4 (وليس layer4 بالكامل)
    لأن layer4 عبارة عن Sequential وقد لا يعطي Grad-Output بالشكل المتوقع دائماً.
    """
    # ResNet عادة: layer4[-1].conv2 أو layer4[-1].conv3 (حسب block)
    if hasattr(model, "layer4"):
        try:
            block = model.layer4[-1]
            # BasicBlock has conv2, Bottleneck has conv3
            if hasattr(block, "conv3"):
                return block.conv3
            if hasattr(block, "conv2"):
                return block.conv2
        except Exception:
            pass

    # fallback: آخر Conv2d موجود في الموديل
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def _to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """
    Colormap شبيه بـ JET بدون matplotlib.
    x: [H,W] in 0..1
    returns uint8 RGB [H,W,3]
    """
    x = np.clip(x, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def gradcam_overlay_base64(
    model: nn.Module,
    pil_img: Image.Image,
    target_class_idx: int,
    device: Optional[torch.device] = None,
    alpha: float = 0.45
) -> str:
    """
    يُرجع صورة overlay بصيغة base64 PNG.
    - لا يعتمد على matplotlib (يحل مشكلة No module named 'matplotlib')
    - أكثر ثباتاً في اختيار طبقة الـConv الأخيرة.
    """
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hooks
    activations = []
    gradients = []

    target_layer = _find_last_conv_layer(model)

    def fwd_hook(_, __, output):
        activations.append(output)

    # full_backward_hook أفضل، ولكن لبعض البيئات قد يفشل -> fallback
    def bwd_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        h2 = target_layer.register_full_backward_hook(bwd_hook)
    except Exception:
        h2 = target_layer.register_backward_hook(bwd_hook)

    try:
        # forward
        x = _transform(pil_img).unsqueeze(0).to(device)
        logits = model(x)

        # تحقّق من حدود target_class_idx
        num_classes = int(logits.shape[1])
        if target_class_idx < 0 or target_class_idx >= num_classes:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        # backward على الكلاس المطلوب
        score = logits[0, target_class_idx]
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            model.zero_grad()
        score.backward(retain_graph=False)

    except Exception:
        # لو أي شيء فشل، رجّع الصورة الأصلية بدون overlay
        return _to_base64_png(pil_img)

    finally:
        # remove hooks
        try:
            h1.remove()
        except Exception:
            pass
        try:
            h2.remove()
        except Exception:
            pass

    if not activations or not gradients:
        return _to_base64_png(pil_img)

    # activations/gradients: [1,C,H,W]
    A = activations[0].detach()[0]   # [C,H,W]
    G = gradients[0].detach()[0]     # [C,H,W]

    # weights: GAP on gradients -> [C]
    weights = G.mean(dim=(1, 2))

    # CAM: sum(w_c * A_c)
    cam = torch.zeros(A.shape[1:], device=A.device)  # [H,W]
    for c in range(A.shape[0]):
        cam += weights[c] * A[c]

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.detach().cpu().numpy()  # [H,W] 0..1

    # resize cam to original size
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0  # 0..1

    # heatmap RGB
    heatmap = _jet_colormap(cam_resized)  # uint8 [H,W,3]

    # original RGB
    orig = np.array(pil_img.convert("RGB")).astype(np.uint8)

    # blend
    a = float(np.clip(alpha, 0.0, 1.0))
    overlay = (orig * (1 - a) + heatmap * a).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)

    return _to_base64_png(overlay_pil)
