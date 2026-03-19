import io
import os
import json
import base64
from typing import List, Optional

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

app = FastAPI(title="Phytologic AI - Phase 1")

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
# تحميل أسماء الفئات
# =========================
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


CLASSES = load_classes()
NUM_CLASSES = len(CLASSES) if CLASSES else 37


# =========================
# التحويلات
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
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
# أدوات مساعدة
# =========================
def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def average_dict_scores(items, key):
    vals = [x.get(key, 0.0) for x in items]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def predict_single_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=2)

    best_idx = int(top_indices[0].item())
    second_idx = int(top_indices[1].item())

    best_conf = float(top_probs[0].item() * 100)
    second_conf = float(top_probs[1].item() * 100)

    best_class = CLASSES[best_idx] if CLASSES else f"class_{best_idx}"
    second_class = CLASSES[second_idx] if CLASSES else f"class_{second_idx}"

    return {
        "best_class": best_class,
        "best_confidence": round(best_conf, 2),
        "second_class": second_class,
        "second_confidence": round(second_conf, 2),
    }


def aggregate_multi_image_predictions(predictions: List[dict]):
    class_scores = {}
    second_candidates = []

    for pred in predictions:
        cls = pred["best_class"]
        class_scores[cls] = class_scores.get(cls, 0.0) + pred["best_confidence"]

        second_candidates.append({
            "class_name": pred["second_class"],
            "confidence": pred["second_confidence"]
        })

    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

    best_class = sorted_classes[0][0]
    best_conf_avg = sum(
        p["best_confidence"] for p in predictions if p["best_class"] == best_class
    ) / max(1, sum(1 for p in predictions if p["best_class"] == best_class))

    # ثاني أفضل تشخيص
    alt_scores = {}
    for pred in predictions:
        for candidate_key in ["best_class", "second_class"]:
            cls = pred[candidate_key]
            conf_key = "best_confidence" if candidate_key == "best_class" else "second_confidence"
            if cls != best_class:
                alt_scores[cls] = alt_scores.get(cls, 0.0) + pred[conf_key]

    if alt_scores:
        second_class = sorted(alt_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        second_conf_avg = alt_scores[second_class] / len(predictions)
    else:
        second_class = best_class
        second_conf_avg = 0.0

    agreement_count = sum(1 for p in predictions if p["best_class"] == best_class)
    agreement_ratio = agreement_count / len(predictions)

    return {
        "best_class": best_class,
        "best_confidence": round(best_conf_avg, 2),
        "second_class": second_class,
        "second_confidence": round(second_conf_avg, 2),
        "agreement_ratio": round(agreement_ratio * 100, 2),
    }


def infer_plant_from_class(class_name: str) -> str:
    if class_name.startswith("Tomato"):
        return "طماطم"
    if class_name.startswith("Potato"):
        return "بطاطس"
    if class_name.startswith("Apple"):
        return "تفاح"
    if class_name.startswith("Grape"):
        return "عنب"
    if class_name.startswith("Corn"):
        return "ذرة"
    if class_name.startswith("Pepper") or class_name.startswith("Bell_pepper"):
        return "فلفل"
    if class_name.startswith("Strawberry"):
        return "فراولة"
    if class_name.startswith("Peach"):
        return "خوخ"
    return "غير محدد"


def infer_disease_name_ar(class_name: str) -> str:
    mapping = {
        "Tomato_Early_blight": "اللفحة المبكرة في الطماطم",
        "Tomato_Septoria_leaf_spot": "تبقع السبتوريا في أوراق الطماطم",
        "Tomato_Bacterial_spot": "التبقع البكتيري في الطماطم",
        "Tomato_Late_blight": "اللفحة المتأخرة في الطماطم",
        "Tomato_Leaf_Mold": "عفن الأوراق في الطماطم",
        "Tomato_Target_Spot": "بقعة الهدف في الطماطم",
        "Tomato_healthy": "نبات سليم",
        "Potato_Early_blight": "اللفحة المبكرة في البطاطس",
        "Potato_Late_blight": "اللفحة المتأخرة في البطاطس",
        "Potato_healthy": "نبات سليم",
    }
    return mapping.get(class_name, class_name)


def build_recommendations(class_name: str):
    recs = {
        "Tomato_Early_blight": [
            "إزالة الأوراق السفلية المصابة",
            "تقليل بلل الأوراق أثناء الري",
            "تحسين التهوية بين النباتات",
            "اتباع برنامج مكافحة مناسب عند الحاجة",
        ],
        "Tomato_Septoria_leaf_spot": [
            "إزالة الأوراق شديدة الإصابة",
            "منع تناثر ماء الري على الأوراق",
            "تحسين التهوية وتقليل الرطوبة",
            "متابعة تطور الإصابة ميدانيًا",
        ],
        "Tomato_Bacterial_spot": [
            "تجنب العمل بين النباتات وهي مبللة",
            "إزالة الأجزاء الشديدة الإصابة",
            "تقليل الرش العلوي بالماء",
            "تعقيم الأدوات المستخدمة",
        ],
    }
    return recs.get(class_name, [
        "افحص الأعراض ميدانيًا",
        "التقط صورًا أوضح عند الحاجة",
        "راجع برنامج المكافحة المناسب للمحصول",
    ])


def build_pesticide_program(class_name: str):
    programs = {
        "Tomato_Early_blight": {
            "material": "استشارة مختص",
            "name": "-",
            "dose": "-",
            "note": "يوصى بالرجوع للمرشد الزراعي أو دليل المبيدات المحلي حسب بلدك."
        },
        "Tomato_Septoria_leaf_spot": {
            "material": "استشارة مختص",
            "name": "-",
            "dose": "-",
            "note": "يفضل اعتماد المبيد الموصى به رسميًا حسب التسجيل المحلي."
        },
        "Tomato_Bacterial_spot": {
            "material": "استشارة مختص",
            "name": "-",
            "dose": "-",
            "note": "الاختيار يعتمد على شدة الإصابة وتوصيات التسجيل المحلي."
        }
    }
    return programs.get(class_name, {
        "material": "استشارة مختص",
        "name": "-",
        "dose": "-",
        "note": "لا يوجد برنامج محدد تلقائيًا لهذه الحالة في المرحلة 1."
    })


# =========================
# نقطة الجذر
# =========================
@app.get("/")
def root():
    return {"message": "Phytologic AI Phase 1 is running"}


# =========================
# التشخيص الاحترافي - المرحلة 1
# =========================
@app.post("/diagnose")
async def diagnose(
    files: List[UploadFile] = File(...),
    farmer_name: str = Form(""),
    farm_name: str = Form(""),
    crop: str = Form(""),
    city: str = Form(""),
    region: str = Form(""),
    latitude: str = Form(""),
    longitude: str = Form(""),
    notes: str = Form("")
):
    if not files or len(files) == 0:
        return {"error": "يرجى رفع صورة واحدة على الأقل"}

    if len(files) > 3:
        return {"error": "الحد الأقصى في المرحلة 1 هو 3 صور"}

    original_images_b64 = []
    per_image_predictions = []
    per_image_quality = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        original_images_b64.append({
            "filename": file.filename,
            "image_b64": image_to_base64(image)
        })

        pred = predict_single_image(image)
        per_image_predictions.append(pred)

        quality = assess_image_quality(image)
        per_image_quality.append(quality)

    agg_pred = aggregate_multi_image_predictions(per_image_predictions)

    avg_quality = average_dict_scores(per_image_quality, "quality_score")
    crop_match = crop_matches_prediction(crop, agg_pred["best_class"])

    decision = build_decision(
        best_confidence=agg_pred["best_confidence"],
        second_confidence=agg_pred["second_confidence"],
        quality_score=avg_quality,
        crop_match=crop_match,
        num_images=len(files)
    )

    plant_name = infer_plant_from_class(agg_pred["best_class"])
    disease_ar = infer_disease_name_ar(agg_pred["best_class"])

    recommendations = build_recommendations(agg_pred["best_class"])
    pesticide_program = build_pesticide_program(agg_pred["best_class"])
    questions = get_questions_for_class(agg_pred["best_class"])

    return {
        "case_data": {
            "farmer_name": farmer_name,
            "farm_name": farm_name,
            "crop": crop,
            "city": city,
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
            "notes": notes,
            "uploaded_images_count": len(files),
        },

        "images": {
            "original_images": original_images_b64
        },

        "quality": {
            "average_quality_score": avg_quality,
            "quality_label": (
                "ممتازة" if avg_quality >= 85 else
                "جيدة" if avg_quality >= 65 else
                "متوسطة" if avg_quality >= 45 else
                "ضعيفة"
            ),
            "per_image_quality": per_image_quality,
        },

        "prediction": {
            "best_prediction": {
                "class_name": agg_pred["best_class"],
                "confidence": agg_pred["best_confidence"],
            },
            "second_prediction": {
                "class_name": agg_pred["second_class"],
                "confidence": agg_pred["second_confidence"],
            },
            "agreement_ratio": agg_pred["agreement_ratio"],
            "confidence_indicator": {
                "value": decision["final_score"],
                "color": decision["decision_color"]
            }
        },

        "crop_validation": {
            "user_crop": crop,
            "predicted_plant": plant_name,
            "match": crop_match
        },

        "final_result": {
            "plant_name": plant_name,
            "disease_name_ar": disease_ar,
            "decision_status": decision["decision_status"],
            "recommended_action": decision["recommended_action"],
            "rejection_reasons": decision["rejection_reasons"]
        },

        "recommendations": recommendations,

        "pesticide_program": pesticide_program,

        "diagnostic_questions": questions,

        "strict_retake_rules": {
            "retake_required": decision["decision_status"] == "غير مؤكد",
            "retake_message": (
                "يرجى إعادة التصوير بصورتين إلى ثلاث صور قريبة وواضحة للورقة المصابة"
                if decision["decision_status"] == "غير مؤكد"
                else "لا حاجة لإعادة التصوير حاليًا"
            ),
            "photo_instructions": [
                "التقط من 2 إلى 3 صور",
                "قرّب الكاميرا من الورقة المصابة",
                "اجعل البقع واضحة داخل الصورة",
                "استخدم إضاءة جيدة",
                "تجنب الظلال والصور المهزوزة"
            ]
        }
    }
