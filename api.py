import io
import os
import json
import base64
from typing import Optional

import torch
import torch.nn as nn
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
# الإعدادات
# =========================================================
MODEL_PATH = "plant_disease_model_v5.pth"
CLASSES_PATH = "classes.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# تحميل أسماء الفئات
# =========================================================
def load_classes():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"لم يتم العثور على {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("ملف classes.json فارغ أو غير صحيح")

    return classes


try:
    CLASSES = load_classes()
    CLASSES_ERROR = ""
except Exception as e:
    CLASSES = []
    CLASSES_ERROR = str(e)
    print("Classes loading error:", e)

NUM_CLASSES = len(CLASSES)


# =========================================================
# تحويل الصور
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
    if NUM_CLASSES == 0:
        raise ValueError("NUM_CLASSES = 0 بسبب مشكلة في classes.json")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
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
    if name.startswith("pepper") or name.startswith("pepper,_bell"):
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
    if name.startswith("soybean"):
        return "صويا"
    if name.startswith("squash"):
        return "اسكواش"
    if name.startswith("orange"):
        return "برتقال"

    return "غير محدد"


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

        "Orange___Haunglongbing_(Citrus_greening)": "التخضير في الحمضيات",

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
        "Tomato___Spider_mites Two-spotted_spider_mite": "إصابة العنكبوت الأحمر في الطماطم",
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
            "إزالة الأجزاء شديدة الإصابة",
            "تعقيم الأدوات",
            "تقليل الرش العلوي"
        ],
        "Tomato___Late_blight": [
            "التخلص من الأوراق شديدة الإصابة",
            "تقليل الرطوبة الحرة على الأوراق",
            "متابعة الحقل يوميًا",
            "التدخل السريع عند زيادة الإصابة"
        ],
        "Potato___Late_blight": [
            "إزالة المجموع الخضري شديد الإصابة",
            "تقليل الرطوبة الحرة على الأوراق",
            "رفع كفاءة الصرف",
            "متابعة الحقل يوميًا"
        ]
    }

    return recs.get(class_name, [
        "افحص الأعراض ميدانيًا",
        "أعد التصوير بصورة أوضح إذا كانت الثقة منخفضة",
        "راجع برنامج المكافحة المناسب للمحصول"
    ])


def check_image_quality(image: Image.Image):
    img = torch.tensor(list(image.getdata()), dtype=torch.float32)
    brightness = float(img.mean().item())

    gray = image.convert("L")
    gray_tensor = torch.tensor(list(gray.getdata()), dtype=torch.float32)
    blur_score = float(gray_tensor.var().item())

    if brightness < 40:
        return False, "الصورة مظلمة جدًا"
    if blur_score < 50:
        return False, "الصورة غير واضحة"
    return True, "جيدة"


def predict_single_image(image: Image.Image):
    if model is None:
        raise RuntimeError(f"الموديل غير جاهز: {MODEL_ERROR}")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    top_probs, top_idxs = torch.topk(probs, 2)

    best_conf = float(top_probs[0][0].item()) * 100
    second_conf = float(top_probs[0][1].item()) * 100

    best_idx = int(top_idxs[0][0].item())
    second_idx = int(top_idxs[0][1].item())

    predicted_class = CLASSES[best_idx]
    second_class = CLASSES[second_idx]

    return {
        "best_class": predicted_class,
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
        "classes_error": CLASSES_ERROR,
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
        "classes_error": CLASSES_ERROR,
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
        if len(files) == 0:
            return {
                "status": "error",
                "message": "يرجى رفع صورة واحدة على الأقل"
            }

        results = []
        images_payload = []

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            is_ok, quality_msg = check_image_quality(image)
            if not is_ok:
                return {
                    "status": "rejected",
                    "reason": quality_msg,
                    "action": "يرجى إعادة التصوير بصورة أوضح"
                }

            pred = predict_single_image(image)
            results.append(pred)

            if return_images:
                images_payload.append({
                    "filename": file.filename,
                    "image_b64": image_to_base64(image)
                })

        main_result = results[0]

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
