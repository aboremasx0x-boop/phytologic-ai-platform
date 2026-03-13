# predict.py
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from gradcam_utils import gradcam_overlay_base64
from severity import estimate_severity_and_recommendations


# سيتم تعبئتها تلقائيًا من ملف الموديل
CLASS_NAMES = []

AR_LABELS = {
    "Bacterial_spot": "التبقّع البكتيري",
    "Early_blight": "اللفحة المبكرة",
    "Late_blight": "اللفحة المتأخرة",
    "Leaf_Mold": "عفن الأوراق",
    "Septoria_leaf_spot": "تبقّع السبتوريا",
    "Target_Spot": "بقعة الهدف",
}

PATHOGEN_TYPE_AR = {
    "Bacterial_spot": "بكتيري",
    "Early_blight": "فطري",
    "Late_blight": "فطري",
    "Leaf_Mold": "فطري",
    "Septoria_leaf_spot": "فطري",
    "Target_Spot": "فطري",
}

PATHOGEN_TYPE_EN = {
    "Bacterial_spot": "bacterial",
    "Early_blight": "fungal",
    "Late_blight": "fungal",
    "Leaf_Mold": "fungal",
    "Septoria_leaf_spot": "fungal",
    "Target_Spot": "fungal",
}


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _torch_load_compat(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)  # type: ignore
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: str, device: Optional[torch.device] = None) -> nn.Module:
    global CLASS_NAMES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj = _torch_load_compat(model_path, device)

    if isinstance(obj, nn.Module):
        model = obj
        model.to(device)
        model.eval()
        return model

    if isinstance(obj, dict):
        if "classes" in obj and isinstance(obj["classes"], list):
            CLASS_NAMES = obj["classes"]

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model_state" in obj and isinstance(obj["model_state"], dict):
            state = obj["model_state"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
        else:
            state = obj

        if not CLASS_NAMES:
            raise ValueError("لم يتم العثور على classes داخل ملف الموديل.")

        model = build_model(num_classes=len(CLASS_NAMES))

        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk.replace("module.", "", 1)
            if nk.startswith("model."):
                nk = nk.replace("model.", "", 1)
            new_state[nk] = v

        model.load_state_dict(new_state, strict=False)
        model.to(device)
        model.eval()
        return model

    raise ValueError("Unsupported model file format.")


_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _softmax_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs


@dataclass
class PredictResult:
    pred_class: str
    pred_label: str
    pathogen_type: str
    confidence: float
    probabilities: Dict[str, float]
    severity: Dict[str, Any]
    gradcam_overlay_b64: str


def predict_image(
    image_bytes: bytes,
    model: nn.Module,
    lang: str = "ar",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    global CLASS_NAMES

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CLASS_NAMES:
        raise RuntimeError("CLASS_NAMES فارغة. تأكد أن ملف الموديل يحتوي على classes.")

    pil_img = _pil_from_bytes(image_bytes)
    x = _transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    probs = _softmax_probs(logits)
    idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    if (lang or "ar").lower().startswith("ar"):
        pred_label = AR_LABELS.get(pred_class, pred_class)
        pathogen_type = PATHOGEN_TYPE_AR.get(pred_class, "غير معروف")
    else:
        pred_label = pred_class
        pathogen_type = PATHOGEN_TYPE_EN.get(pred_class, "unknown")

    severity = estimate_severity_and_recommendations(
        pil_img=pil_img,
        pred_class=pred_class,
        lang=lang
    )

    gradcam_b64 = gradcam_overlay_base64(
        model=model,
        pil_img=pil_img,
        target_class_idx=idx,
        device=device
    )

    return {
        "pred_class": pred_class,
        "pred_label": pred_label,
        "pathogen_type": pathogen_type,
        "confidence": confidence,
        "probabilities": probabilities,
        "severity": severity,
        "recommendations": severity.get("recommendations", []),
        "gradcam_overlay_b64": gradcam_b64,
    }
