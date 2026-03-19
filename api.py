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

MODEL_PATH = "plant_disease_model_v4.pth"
DEVICE = torch.device("cpu")
NUM_CLASSES = 37


def load_classes():
    if os.path.exists("classes.json"):
        with open("classes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return [f"class_{i}" for i in range(NUM_CLASSES)]


CLASSES = load_classes()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


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
    return "غير محدد"


def infer_disease_name_ar(class_name: str) -> str:
    mapping = {
        "Tomato_Early_blight": "اللفحة المبكرة في الطماطم",
        "Tomato_Septoria_leaf_spot": "تبقع السبتوريا في أوراق الطماطم",
        "Tomato_Bacterial_spot": "التبقع البكتيري في الطماطم",
        "Tomato_Late_blight": "اللفحة المتأخرة في الطماطم",
        "Tomato_Leaf_Mold": "عفن الأوراق في الطماطم",
        "Tomato_Target_Spot": "بقعة الهدف في الطماطم",
        "Tomato_healthy": "طماطم سليمة",
        "Potato_Early_blight": "اللفحة المبكرة في البطاطس",
        "Potato_Late_blight": "اللفحة المتأخرة في البطاطس",
        "Potato_healthy": "بطاطس سليمة",
        "Apple_Scab": "جرب التفاح",
        "Apple_rust": "صدأ التفاح",
        "Apple_healthy": "تفاح سليم",
        "Grape_Black_rot": "العفن الأسود في العنب",
        "Grape_healthy": "عنب سليم",
    }
    return mapping.get(class_name, class_name)


def build_recommendations(class_name: str):
    recs = {
        "Tomato_Early_blight": [
            "إزالة الأوراق السفلية المصابة",
            "تقليل ملامسة الماء للأوراق",
            "تحسين التهوية بين النباتات",
            "متابعة تطور الإصابة خلال الأيام القادمة",
        ],
        "Tomato_Septoria_leaf_spot": [
            "إزالة الأوراق شديدة الإصابة",
            "تقليل الرطوبة على الأوراق",
            "تحسين التهوية",
            "إعادة التصوير إذا زادت البقع",
        ],
        "Tomato_Bacterial_spot": [
            "تجنب لمس النباتات وهي مبللة",
            "إزالة الأوراق شديدة الإصابة",
            "تعقيم الأدوات",
            "تقليل الرش العلوي بالماء",
        ],
    }
    return recs.get(class_name, [
        "افحص الأعراض ميدانيًا",
        "التقط صورًا أوضح عند الحاجة",
        "راجع برنامج المكافحة المناسب للمحصول",
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

    best_class = CLASSES[best_idx]
    second_class = CLASSES[second_idx]

    return {
        "best_class": best_class,
        "best_confidence": round(best_conf, 2),
        "second_class": second_class,
        "second_confidence": round(second_conf, 2),
    }


@app.get("/")
def root():
    return {
        "status": "running",
        "model_ready": MODEL_READY,
        "model_path": MODEL_PATH,
        "num_classes": NUM_CLASSES,
        "model_error": MODEL_ERROR
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_READY else "model_error",
        "model_loaded": MODEL_READY,
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

        # نأخذ أول صورة الآن كملخص مبدئي
        main_result = results[0]

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
