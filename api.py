import io
import os
import json
import base64
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models

from image_quality import assess_image_quality
from disease_rules import (
    crop_matches_prediction,
    get_questions_for_class,
    build_decision,
)

app = FastAPI(title="Phytologic AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "plant_disease_model_v5.pth"
CLASSES_PATH = "classes.json"
DEVICE = torch.device("cpu")


# =========================
# تحميل الفئات
# =========================
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


CLASSES = load_classes()
NUM_CLASSES = len(CLASSES) if CLASSES else 38


# =========================
# تحويل الصور
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================
# تحميل الموديل
# =========================
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


model = load_model()


# =========================
# أدوات
# =========================
def image_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict(image: Image.Image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idx = torch.topk(probs, 2)

    return {
        "best": CLASSES[top_idx[0][0]],
        "best_conf": float(top_probs[0][0] * 100),
        "second": CLASSES[top_idx[0][1]],
        "second_conf": float(top_probs[0][1] * 100),
    }


# =========================
# API
# =========================
@app.get("/")
def root():
    return {"status": "running"}


@app.post("/diagnose")
async def diagnose(
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
    crop: str = Form(""),
    city: str = Form("")
):
    try:
        files = [f for f in [file1, file2, file3] if f]

        images_b64 = []
        results = []
        qualities = []

        for file in files:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")

            images_b64.append(image_to_base64(img))

            pred = predict(img)
            results.append(pred)

            q = assess_image_quality(img)
            qualities.append(q["quality_score"])

        best = results[0]

        return {
            "status": "success",
            "diagnosis_1": best["best"],
            "confidence_1": round(best["best_conf"], 2),
            "diagnosis_2": best["second"],
            "confidence_2": round(best["second_conf"], 2),
            "quality_avg": round(sum(qualities)/len(qualities), 2),
            "images": images_b64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
