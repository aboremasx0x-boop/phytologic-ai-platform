import io
import os
import json
import base64
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models

app = FastAPI(title="Phytologic AI Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# الإعدادات
# =========================================
MODEL_PATH = "plant_disease_model_v5.pth"
CLASSES_PATH = "classes.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# تحميل الكلاسات
# =========================================
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"لم يتم العثور على {CLASSES_PATH}")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

NUM_CLASSES = len(CLASSES)

# =========================================
# بناء الموديل
# =========================================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# آخر طبقة convolution لاستخدام Grad-CAM
TARGET_LAYER = model.features[-1]

# =========================================
# التحويل
# =========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================================
# أدوات مساعدة
# =========================================
def confidence_label(conf: float) -> str:
    if conf >= 90:
        return "مرتفعة جدًا"
    if conf >= 80:
        return "مرتفعة"
    if conf >= 70:
        return "متوسطة"
    return "منخفضة"


def infer_plant(class_name: str) -> str:
    return class_name.split("___")[0] if "___" in class_name else class_name


def infer_disease_name_ar(class_name: str) -> str:
    mapping = {
        "Apple___Apple_scab": "جرب التفاح",
        "Apple___Black_rot": "العفن الأسود في التفاح",
        "Apple___Cedar_apple_rust": "صدأ التفاح",
        "Apple___healthy": "تفاح سليم",
        "Blueberry___healthy": "توت أزرق سليم",
        "Cherry___Powdery_mildew": "البياض الدقيقي في الكرز",
        "Cherry___healthy": "كرز سليم",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "التبقع الرمادي في الذرة",
        "Corn___Common_rust": "صدأ الذرة",
        "Corn___Northern_Leaf_Blight": "لفحة أوراق الذرة الشمالية",
        "Corn___healthy": "ذرة سليمة",
        "Grape___Black_rot": "العفن الأسود في العنب",
        "Grape___Esca_(Black_Measles)": "إسكا العنب",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "لفحة أوراق العنب",
        "Grape___healthy": "عنب سليم",
        "Orange___Haunglongbing_(Citrus_greening)": "تخضير الحمضيات",
        "Peach___Bacterial_spot": "التبقع البكتيري في الخوخ",
        "Peach___healthy": "خوخ سليم",
        "Pepper,_bell___Bacterial_spot": "التبقع البكتيري في الفلفل",
        "Pepper,_bell___healthy": "فلفل سليم",
        "Potato___Early_blight": "اللفحة المبكرة في البطاطس",
        "Potato___Late_blight": "اللفحة المتأخرة في البطاطس",
        "Potato___healthy": "بطاطس سليمة",
        "Raspberry___healthy": "رازبيري سليم",
        "Soybean___healthy": "صويا سليمة",
        "Squash___Powdery_mildew": "البياض الدقيقي في الاسكواش",
        "Strawberry___Leaf_scorch": "لفحة أوراق الفراولة",
        "Strawberry___healthy": "فراولة سليمة",
        "Tomato___Bacterial_spot": "التبقع البكتيري في الطماطم",
        "Tomato___Early_blight": "اللفحة المبكرة في الطماطم",
        "Tomato___Late_blight": "اللفحة المتأخرة في الطماطم",
        "Tomato___Leaf_Mold": "عفن أوراق الطماطم",
        "Tomato___Septoria_leaf_spot": "تبقع السبتوريا في أوراق الطماطم",
        "Tomato___Spider_mites Two-spotted_spider_mite": "العنكبوت الأحمر في الطماطم",
        "Tomato___Target_Spot": "بقعة الهدف في الطماطم",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "فيروس تجعد واصفرار أوراق الطماطم",
        "Tomato___Tomato_mosaic_virus": "فيروس موزاييك الطماطم",
        "Tomato___healthy": "طماطم سليمة",
    }
    return mapping.get(class_name, class_name)


def build_recommendations(class_name: str):
    recs = {
        "Tomato___Early_blight": [
            "إزالة الأوراق السفلية المصابة",
            "تقليل بلل الأوراق أثناء الري",
            "تحسين التهوية بين النباتات",
            "متابعة برنامج المكافحة المناسب"
        ],
        "Tomato___Septoria_leaf_spot": [
            "إزالة الأوراق شديدة الإصابة",
            "تقليل الرطوبة على الأوراق",
            "تحسين التهوية",
            "إعادة التصوير إذا زادت الأعراض"
        ],
        "Tomato___Bacterial_spot": [
            "تجنب لمس النباتات وهي مبللة",
            "تعقيم الأدوات الزراعية",
            "تقليل الرش العلوي",
            "إزالة الأجزاء شديدة الإصابة"
        ],
        "Tomato___Late_blight": [
            "التخلص من الأوراق شديدة الإصابة",
            "تقليل الرطوبة الحرة على الأوراق",
            "المتابعة اليومية",
            "التدخل السريع عند زيادة الأعراض"
        ],
    }
    return recs.get(class_name, [
        "افحص الأعراض ميدانيًا",
        "أعد التصوير بصورة أوضح إذا كانت الثقة منخفضة",
        "راجع برنامج المكافحة المناسب للمحصول"
    ])


def to_base64_pil(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def check_image_quality(image: Image.Image):
    arr = np.array(image).astype(np.float32)
    brightness = float(arr.mean())

    gray = np.array(image.convert("L")).astype(np.float32)
    blur_score = float(gray.var())

    if brightness < 40:
        return False, "الصورة مظلمة جدًا"
    if blur_score < 50:
        return False, "الصورة غير واضحة"
    return True, "جيدة"


def smart_decision(best_conf: float, second_conf: float):
    diff = best_conf - second_conf

    if best_conf < 50:
        decision = "غير مؤكد"
    elif diff < 5:
        decision = "تشابه عالي"
    elif diff < 10:
        decision = "تشابه متوسط"
    else:
        decision = "تشخيص واضح"

    return decision, round(diff, 2)


# =========================================
# Grad-CAM
# =========================================
def generate_gradcam(image: Image.Image):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = TARGET_LAYER.register_forward_hook(forward_hook)
    bh = TARGET_LAYER.register_full_backward_hook(backward_hook)

    try:
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        image_tensor.requires_grad_(True)

        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        model.zero_grad()
        score = outputs[0, pred_idx.item()]
        score.backward()

        act = activations[0]           # [1, C, H, W]
        grad = gradients[0]            # [1, C, H, W]

        weights = grad.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        return cam, pred_idx.item(), float(confidence.item()) * 100

    finally:
        fh.remove()
        bh.remove()


def colorize_heatmap(cam_2d: np.ndarray, out_size):
    h, w = out_size[1], out_size[0]
    cam_img = Image.fromarray(np.uint8(cam_2d * 255)).resize((w, h))
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0

    # heatmap بسيط: أحمر قوي، أخضر أقل، أزرق صفر
    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[..., 0] = np.uint8(cam_arr * 255)          # R
    heat[..., 1] = np.uint8(cam_arr * 140)          # G
    heat[..., 2] = 0                                # B
    return heat


def overlay_gradcam(original_image: Image.Image, cam_2d: np.ndarray, alpha=0.4):
    original = original_image.resize(original_image.size).convert("RGB")
    orig_arr = np.array(original).astype(np.float32)

    heat = colorize_heatmap(cam_2d, original.size).astype(np.float32)

    blended = (orig_arr * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def estimate_severity_from_cam(cam_2d: np.ndarray):
    # تقدير تقريبي لنسبة الإصابة اعتمادًا على المساحات النشطة
    mask = cam_2d >= 0.45
    severity_pct = float(mask.mean() * 100)

    if severity_pct < 5:
        label = "منخفضة"
    elif severity_pct < 15:
        label = "متوسطة"
    else:
        label = "مرتفعة"

    return round(severity_pct, 2), label


# =========================================
# التنبؤ
# =========================================
def predict_single_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idxs = torch.topk(probs, 2)

    best_conf = float(top_probs[0][0].item()) * 100
    second_conf = float(top_probs[0][1].item()) * 100

    best_idx = int(top_idxs[0][0].item())
    second_idx = int(top_idxs[0][1].item())

    predicted_class = CLASSES[best_idx]
    second_class = CLASSES[second_idx]

    decision, diff = smart_decision(best_conf, second_conf)

    return {
        "best_class": predicted_class,
        "best_confidence": round(best_conf, 2),
        "second_class": second_class,
        "second_confidence": round(second_conf, 2),
        "confidence_diff": diff,
        "decision": decision
    }


# =========================================
# Routes
# =========================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "classes_loaded": len(CLASSES),
        "device": str(DEVICE),
        "classes_error": "",
        "model_error": ""
    }


@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    return_image: bool = Form(False),
    return_gradcam: bool = Form(True)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        is_ok, quality_msg = check_image_quality(image)
        if not is_ok:
            return {
                "status": "rejected",
                "reason": quality_msg,
                "action": "يرجى إعادة التصوير بصورة أوضح"
            }

        result = predict_single_image(image)

        if result["best_confidence"] < 50:
            response = {
                "status": "uncertain",
                "message": "التشخيص غير مؤكد",
                "confidence": result["best_confidence"],
                "best_prediction": {
                    "class_name": result["best_class"],
                    "disease_name_ar": infer_disease_name_ar(result["best_class"]),
                    "plant_name": infer_plant(result["best_class"])
                },
                "decision_analysis": {
                    "decision": result["decision"],
                    "confidence_diff": result["confidence_diff"]
                },
                "action": "يرجى إعادة التصوير بصورة أوضح أو من زاوية مختلفة"
            }
            return response

        response = {
            "status": "success",
            "prediction": {
                "class_name": result["best_class"],
                "disease_name_ar": infer_disease_name_ar(result["best_class"]),
                "plant_name": infer_plant(result["best_class"]),
                "confidence": result["best_confidence"],
                "confidence_label": confidence_label(result["best_confidence"])
            },
            "second_prediction": {
                "class_name": result["second_class"],
                "disease_name_ar": infer_disease_name_ar(result["second_class"]),
                "plant_name": infer_plant(result["second_class"]),
                "confidence": result["second_confidence"],
                "confidence_label": confidence_label(result["second_confidence"])
            },
            "decision_analysis": {
                "decision": result["decision"],
                "confidence_diff": result["confidence_diff"]
            },
            "recommendations": build_recommendations(result["best_class"])
        }

        if return_gradcam:
            cam_2d, _, _ = generate_gradcam(image)
            gradcam_overlay = overlay_gradcam(image, cam_2d, alpha=0.4)
            severity_pct, severity_label = estimate_severity_from_cam(cam_2d)

            response["gradcam"] = {
                "overlay_b64": to_base64_pil(gradcam_overlay),
                "severity_percent": severity_pct,
                "severity_label": severity_label
            }

        if return_image:
            response["original_image_b64"] = to_base64_pil(image)

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
