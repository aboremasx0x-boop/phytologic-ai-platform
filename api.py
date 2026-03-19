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

# =========================================================
# إعدادات ثابتة
# =========================================================
MODEL_PATH = "plant_disease_model_v4.pth"
CLASSES_PATH = "classes.json"
DEVICE = torch.device("cpu")
NUM_CLASSES = 37


# =========================================================
# تحميل أسماء الأمراض
# =========================================================
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)
            if isinstance(classes, list) and len(classes) > 0:
                return classes
    return [f"class_{i}" for i in range(NUM_CLASSES)]


CLASSES = load_classes()


# =========================================================
# تحويل الصورة
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# تحميل الموديل
# =========================================================
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
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as e:
    model = None
    MODEL_READY = False
    MODEL_ERROR = str(e)
    print("Model loading error:", e)


# =========================================================
# أدوات مساعدة
# =========================================================
def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def confidence_label(conf: float) -> str:
    if conf >= 90:
        return "مرتفعة جدًا"
    if conf >= 80:
        return "مرتفعة"
    if conf >= 70:
        return "متوسطة"
    return "منخفضة"


def infer_plant_from_class(class_name: str) -> str:
    name = class_name.lower()

    if name.startswith("tomato"):
        return "طماطم"
    if name.startswith("potato"):
        return "بطاطس"
    if name.startswith("apple"):
        return "تفاح"
    if name.startswith("grape"):
        return "عنب"
    if name.startswith("corn"):
        return "ذرة"
    if name.startswith("pepper") or name.startswith("bell_pepper"):
        return "فلفل"
    if name.startswith("strawberry"):
        return "فراولة"
    if name.startswith("peach"):
        return "خوخ"
    if name.startswith("blueberry"):
        return "توت أزرق"
    if name.startswith("cherry"):
        return "كرز"
    if name.startswith("raspberry"):
        return "رازبيري"
    if name.startswith("soybean") or name.startswith("soyabean"):
        return "صويا"
    if name.startswith("squash"):
        return "اسكواش"

    return "غير محدد"


def infer_disease_name_ar(class_name: str) -> str:
    mapping = {
        "Tomato___Early_blight": "اللفحة المبكرة في الطماطم",
        "Tomato___Septoria_leaf_spot": "تبقع السبتوريا في أوراق الطماطم",
        "Tomato___Bacterial_spot": "التبقع البكتيري في الطماطم",
        "Tomato___Late_blight": "اللفحة المتأخرة في الطماطم",
        "Tomato___Leaf_Mold": "عفن الأوراق في الطماطم",
        "Tomato___Target_Spot": "بقعة الهدف في الطماطم",
        "Tomato___healthy": "طماطم سليمة",
        "Tomato___Spider_mites Two-spotted_spider_mite": "إصابة حلم العنكبوت الأحمر في الطماطم",
        "Tomato___Tomato_mosaic_virus": "فيروس موزاييك الطماطم",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "فيروس تجعد واصفرار أوراق الطماطم",
        "Potato___Early_blight": "اللفحة المبكرة في البطاطس",
        "Potato___Late_blight": "اللفحة المتأخرة في البطاطس",
        "Potato___healthy": "بطاطس سليمة",
        "Apple___Apple_scab": "جرب التفاح",
        "Apple___Cedar_apple_rust": "صدأ التفاح",
        "Apple___healthy": "تفاح سليم",
        "Grape___Black_rot": "العفن الأسود في العنب",
        "Grape___Esca_(Black_Measles)": "إسكا العنب",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "لفحة أوراق العنب",
        "Grape___healthy": "عنب سليم",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "التبقع الرمادي في الذرة",
        "Corn___Common_rust": "صدأ الذرة",
        "Corn___Northern_Leaf_Blight": "لفحة أوراق الذرة الشمالية",
        "Corn___healthy": "ذرة سليمة",
        "Pepper,_bell___Bacterial_spot": "التبقع البكتيري في الفلفل",
        "Pepper,_bell___healthy": "فلفل سليم",
        "Strawberry___Leaf_scorch": "لفحة أوراق الفراولة",
        "Strawberry___healthy": "فراولة سليمة",
        "Peach___Bacterial_spot": "التبقع البكتيري في الخوخ",
        "Peach___healthy": "خوخ سليم",
        "Blueberry___healthy": "توت أزرق سليم",
        "Cherry___Powdery_mildew": "البياض الدقيقي في الكرز",
        "Cherry___healthy": "كرز سليم",
        "Raspberry___healthy": "رازبيري سليم",
        "Soybean___healthy": "صويا سليمة",
        "Squash___Powdery_mildew": "البياض الدقيقي في الاسكواش",
        "Orange___Haunglongbing_(Citrus_greening)": "التخضير في الحمضيات",
    }
    return mapping.get(class_name, class_name)


def build_recommendations(class_name: str):
    recs = {
        "Tomato___Early_blight": [
            "إزالة الأوراق السفلية المصابة",
            "تقليل ملامسة الماء للأوراق",
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
            "تجنب لمس النبات وهو مبلل",
            "إزالة الأجزاء شديدة الإصابة",
            "تعقيم الأدوات",
            "تقليل الرش العلوي"
        ],
        "Tomato___Late_blight": [
            "التخلص من الأوراق شديدة الإصابة",
            "تقليل الرطوبة الحرة على الأوراق",
            "متابعة الحقل بشكل يومي",
            "التدخل السريع عند زيادة الأعراض"
        ]
    }
    return recs.get(class_name, [
        "افحص الأعراض ميدانيًا",
        "التقط صورة أوضح إذا كانت الثقة منخفضة",
        "راجع برنامج المكافحة المناسب للمحصول"
    ])


def predict_single_image(image: Image.Image):
    if model is None:
        raise RuntimeError(f"الموديل غير جاهز: {MODEL_ERROR}")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=2)

    best_idx = int(top_indices[0].item())
    second_idx = int(top_indices[1].item())

    best_conf = float(top_probs[0].item() * 100)
    second_conf = float(top_probs[1].item() * 100)

    best_class = CLASSES[best_idx] if best_idx < len(CLASSES) else f"class_{best_idx}"
    second_class = CLASSES[second_idx] if second_idx < len(CLASSES) else f"class_{second_idx}"

    return {
        "best_class": best_class,
        "best_confidence": round(best_conf, 2),
        "second_class": second_class,
        "second_confidence": round(second_conf, 2),
    }


# =========================================================
# المسارات
# =========================================================
@app.get("/")
def root():
    return {
        "status": "running",
        "model_ready": MODEL_READY,
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "classes_loaded": len(CLASSES),
        "model_error": MODEL_ERROR
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY else "model_error",
        "model_loaded": MODEL_READY,
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "classes_loaded": len(CLASSES),
        "device": str(DEVICE),
        "model_error": MODEL_ERROR
    }


@app.post("/diagnose")
async def diagnose(
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
    crop: str = Form(""),
    city: str = Form(""),
    return_images: bool = Form(False)
):
    try:
        if model is None:
            return {
                "status": "error",
                "message": f"الموديل غير جاهز: {MODEL_ERROR}"
            }

        files = [f for f in [file1, file2, file3] if f is not None]
        results = []
        images_payload = []

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            pred = predict_single_image(image)
            results.append(pred)

            if return_images:
                images_payload.append({
                    "filename": file.filename,
                    "image_b64": image_to_base64(image)
                })

        main_result = results[0]

        # قاعدة رفض ذكية إذا كانت الثقة منخفضة
        if main_result["best_confidence"] < 50:
            response = {
                "status": "uncertain",
                "message": "التشخيص غير مؤكد",
                "confidence": main_result["best_confidence"],
                "best_prediction": {
                    "class_name": main_result["best_class"],
                    "disease_name_ar": infer_disease_name_ar(main_result["best_class"]),
                    "plant_name": infer_plant_from_class(main_result["best_class"])
                },
                "action": "يرجى إعادة التصوير بصورة أوضح أو من زاوية مختلفة أو استخدام 2 إلى 3 صور"
            }

            if return_images:
                response["images"] = images_payload

            return response

        response = {
            "status": "success",
            "uploaded_images_count": len(files),
            "crop": crop,
            "city": city,
            "best_prediction": {
                "class_name": main_result["best_class"],
                "disease_name_ar": infer_disease_name_ar(main_result["best_class"]),
                "plant_name": infer_plant_from_class(main_result["best_class"]),
                "confidence": main_result["best_confidence"],
                "confidence_label": confidence_label(main_result["best_confidence"]),
            },
            "second_prediction": {
                "class_name": main_result["second_class"],
                "disease_name_ar": infer_disease_name_ar(main_result["second_class"]),
                "plant_name": infer_plant_from_class(main_result["second_class"]),
                "confidence": main_result["second_confidence"],
                "confidence_label": confidence_label(main_result["second_confidence"]),
            },
            "recommendations": build_recommendations(main_result["best_class"]),
            "all_results": results
        }

        if return_images:
            response["images"] = images_payload

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"حدث خطأ أثناء التشخيص: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
