import io
import os
import json
import base64
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models

app = FastAPI(title="Phytologic AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== إعدادات أساسية ======
MODEL_PATH = "plant_disease_model_v4.pth"
DEVICE = torch.device("cpu")

# ❗ الحل الحاسم هنا
NUM_CLASSES = 37


# ====== تحويل الصورة ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ====== تحميل الموديل ======
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


try:
    model = load_model()
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print("❌ Model loading error:", e)


# ====== تحويل صورة Base64 ======
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ====== التنبؤ ======
def predict(image):
    if model is None:
        raise RuntimeError("Model not loaded")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idxs = torch.topk(probs, 2)

    return {
        "top1_class": int(top_idxs[0][0]),
        "top1_conf": float(top_probs[0][0] * 100),
        "top2_class": int(top_idxs[0][1]),
        "top2_conf": float(top_probs[0][1] * 100),
    }


# ====== المسارات ======

@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES
    }


@app.post("/diagnose")
async def diagnose(
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
):
    try:
        files = [f for f in [file1, file2, file3] if f]

        results = []
        images_b64 = []

        for f in files:
            img_bytes = await f.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            pred = predict(img)
            results.append(pred)

            images_b64.append(image_to_base64(img))

        return {
            "status": "success",
            "predictions": results,
            "images": images_b64
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ====== تشغيل Render ======
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
