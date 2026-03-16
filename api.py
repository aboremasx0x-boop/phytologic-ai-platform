import io
import os
import uuid
import base64
import json
import csv
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse
from torchvision import transforms, models

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from apscheduler.schedulers.background import BackgroundScheduler

import arabic_reshaper
from bidi.algorithm import get_display

from disease_info import DISEASE_INFO
from database import init_db, save_diagnosis, save_alert, save_farmer, get_connection
from ai_forecast_service import AIForecastService
from sms_service import SMSService


# =========================
# أدوات أساسية
# =========================

def get_db_connection():
    return get_connection()


def extract_disease_region(image_pil):
    img = np.array(image_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([10, 50, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image_pil

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    crop = img[y:y+h, x:x+w]
    crop = cv2.resize(crop, (224, 224))

    return Image.fromarray(crop)


# =========================
# إنشاء التطبيق
# =========================

app = FastAPI(title="Phytologic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


# =========================
# الصفحات الرئيسية
# =========================

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/pages/{page_name}")
def open_page(page_name: str):
    file_path = f"{page_name}.html" if not page_name.endswith(".html") else page_name

    if os.path.exists(file_path):
        return FileResponse(file_path)

    return JSONResponse({"error": "page not found"}, status_code=404)




   

# =========================
# تهيئة الخدمات
# =========================

forecast_ai_service = AIForecastService()

sms_service = SMSService(
    app_sid="",
    sender="Phytologic"
)


# =========================
# تهيئة قاعدة البيانات + التخزين
# =========================

DATA_DIR = os.getenv("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

init_db()

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================
# إعدادات نموذج الذكاء الاصطناعي
# =========================

MODEL_PATH = "plant_disease_model_v3.pth"
IMG_SIZE = 160

if not os.path.exists(MODEL_PATH):
    url = "https://github.com/aboremasx0x-boop/phytologic-ai-platform/releases/download/v1/plant_disease_model_v3.pth"
    urllib.request.urlretrieve(url, MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes = checkpoint["classes"]
num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# =========================
# دعم الخط العربي في PDF
# =========================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(BASE_PATH, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

def ensure_font_file(local_path: str, download_url: str):
    if not os.path.exists(local_path):
        try:
            urllib.request.urlretrieve(download_url, local_path)
            print(f"Downloaded font: {local_path}")
        except Exception as e:
            print(f"Failed to download font {local_path}: {e}")

# حمّل الخط تلقائيًا إذا لم يكن موجودًا
ensure_font_file(
    os.path.join(FONTS_DIR, "NotoNaskhArabic-Regular.ttf"),
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNaskhArabic/NotoNaskhArabic-Regular.ttf"
)

ensure_font_file(
    os.path.join(FONTS_DIR, "Amiri-Regular.ttf"),
    "https://github.com/aliftype/amiri/raw/master/fonts/ttf/Amiri-Regular.ttf"
)

AR_FONT_REGISTERED = False
AR_FONT_PATH_USED = None

for font_path in [
    os.path.join(FONTS_DIR, "NotoNaskhArabic-Regular.ttf"),
    os.path.join(FONTS_DIR, "Amiri-Regular.ttf"),
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf"
]:
    print("Checking font path:", font_path, "=>", os.path.exists(font_path))
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("ARABIC_FONT", font_path))
            AR_FONT_REGISTERED = True
            AR_FONT_PATH_USED = font_path
            print("Arabic font registered successfully:", font_path)
            break
        except Exception as e:
            print("Font registration failed:", font_path, e)

print("AR_FONT_REGISTERED =", AR_FONT_REGISTERED)
print("AR_FONT_PATH_USED =", AR_FONT_PATH_USED)
print("CURRENT_WORKING_DIR =", os.getcwd())
print("BASE_PATH =", BASE_PATH)
# =========================
# إحداثيات المناطق والمدن
# =========================

REGION_COORDS = {
    "الرياض": {"lat": 24.7136, "lon": 46.6753, "city": "الرياض"},
    "مكة": {"lat": 21.3891, "lon": 39.8579, "city": "جدة"},
    "المدينة": {"lat": 24.5247, "lon": 39.5692, "city": "المدينة المنورة"},
    "القصيم": {"lat": 26.3592, "lon": 43.9818, "city": "بريدة"},
    "حائل": {"lat": 27.5114, "lon": 41.7208, "city": "حائل"},
    "تبوك": {"lat": 28.3998, "lon": 36.5715, "city": "تبوك"},
    "الجوف": {"lat": 29.9697, "lon": 40.2064, "city": "سكاكا"},
    "الحدود الشمالية": {"lat": 30.9753, "lon": 41.0381, "city": "عرعر"},
    "الشرقية": {"lat": 26.4207, "lon": 50.0888, "city": "الدمام"},
    "الباحة": {"lat": 20.0129, "lon": 41.4677, "city": "الباحة"},
    "عسير": {"lat": 18.2164, "lon": 42.5053, "city": "أبها"},
    "جازان": {"lat": 16.8892, "lon": 42.5511, "city": "جازان"},
    "نجران": {"lat": 17.5650, "lon": 44.2289, "city": "نجران"}
}

CITY_COORDS = {
    "الرياض": (24.7136, 46.6753),
    "جدة": (21.4858, 39.1925),
    "مكة": (21.3891, 39.8579),
    "المدينة": (24.5247, 39.5692),
    "المدينة المنورة": (24.5247, 39.5692),
    "بريدة": (26.3592, 43.9818),
    "حائل": (27.5114, 41.7208),
    "تبوك": (28.3998, 36.5715),
    "سكاكا": (29.9697, 40.2064),
    "عرعر": (30.9753, 41.0381),
    "الدمام": (26.4207, 50.0888),
    "الباحة": (20.0129, 41.4677),
    "أبها": (18.2164, 42.5053),
    "جازان": (16.8892, 42.5511),
    "نجران": (17.5650, 44.2289)
}

# =========================
# أدوات مساعدة عامة
# =========================

def fix_arabic(text: str) -> str:
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr)

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)





def save_upload_file(image_bytes: bytes, original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1].lower().strip()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path


def fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PhytologicAI/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_region_or_city_coords(region=None, city=None, latitude=None, longitude=None):
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude), "direct"

    region = (region or "").strip()
    city = (city or "").strip()

    if city and city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        return lat, lon, "city"

    if region and region in REGION_COORDS:
        return REGION_COORDS[region]["lat"], REGION_COORDS[region]["lon"], "region"

    return 24.7136, 46.6753, "fallback"
# =========================
# قاعدة بيانات المبيدات
# =========================

PESTICIDE_DATABASE = {
    "Tomato_Early_blight": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Chlorothalonil",
            "trade_name": "Bravo",
            "type": "fungicide",
            "dose": "2 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Mancozeb",
            "trade_name": "Dithane M-45",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Azoxystrobin",
            "trade_name": "Amistar",
            "type": "fungicide",
            "dose": "1 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        }
    ],
    "Tomato_Late_blight": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Metalaxyl + Mancozeb",
            "trade_name": "Ridomil Gold",
            "type": "fungicide",
            "dose": "2.5 جم لكل لتر ماء",
            "phi": "10 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Copper oxychloride",
            "trade_name": "Kocide",
            "type": "fungicide",
            "dose": "2.5 جم لكل لتر ماء",
            "phi": "10 أيام",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Cymoxanil + Mancozeb",
            "trade_name": "Curzate",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        }
    ],
    "Tomato_Leaf_Mold": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Chlorothalonil",
            "trade_name": "Bravo",
            "type": "fungicide",
            "dose": "2 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Mancozeb",
            "trade_name": "Dithane M-45",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Copper oxychloride",
            "trade_name": "Kocide",
            "type": "fungicide",
            "dose": "2.5 جم لكل لتر ماء",
            "phi": "10 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        }
    ],
    "Grape_Black_rot": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Mancozeb",
            "trade_name": "Dithane M-45",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Azoxystrobin",
            "trade_name": "Amistar",
            "type": "fungicide",
            "dose": "1 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Copper hydroxide",
            "trade_name": "Kocide 2000",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "10 أيام",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        }
    ],
    "Corn_Blight": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Azoxystrobin",
            "trade_name": "Amistar",
            "type": "fungicide",
            "dose": "1 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Propiconazole",
            "trade_name": "Tilt",
            "type": "fungicide",
            "dose": "0.5 مل لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Mancozeb",
            "trade_name": "Dithane M-45",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        }
    ],
    "Powdery_Mildew": [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Sulfur",
            "trade_name": "Thiovit",
            "type": "fungicide",
            "dose": "3 جم لكل لتر ماء",
            "phi": "5 أيام",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Myclobutanil",
            "trade_name": "Systhane",
            "type": "fungicide",
            "dose": "0.4 مل لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Triadimenol",
            "trade_name": "Bayfidan",
            "type": "fungicide",
            "dose": "0.5 مل لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        }
    ]
}


def normalize_pesticide_key(best_class: str, cause: str = "") -> str:
    name = (best_class or "").replace("-", "_").replace(" ", "_")
    low = name.lower()
    cause_low = (cause or "").lower()

    if "healthy" in low:
        return "HEALTHY"
    if "tomato" in low and "early" in low and "blight" in low:
        return "Tomato_Early_blight"
    if "tomato" in low and "late" in low and "blight" in low:
        return "Tomato_Late_blight"
    if "leaf" in low and "mold" in low and "tomato" in low:
        return "Tomato_Leaf_Mold"
    if "grape" in low and ("black" in low or "rot" in low):
        return "Grape_Black_rot"
    if "corn" in low and ("blight" in low or "leaf_blight" in low or "leaf" in low):
        return "Corn_Blight"
    if "powdery" in low and "mildew" in low:
        return "Powdery_Mildew"
    if "fungal" in cause_low or "فطري" in cause:
        return "Powdery_Mildew"

    return "GENERIC"


def get_default_pesticide_program(best_class: str, cause: str = "", severity_level: str = "moderate") -> list:
    low = (best_class or "").lower()

    if "healthy" in low:
        return [{
            "title_ar": "لا يحتاج مبيد",
            "active_ingredient": "لا يوجد",
            "trade_name": "-",
            "type": "لا حاجة للعلاج الكيميائي",
            "dose": "-",
            "phi": "-",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "no_spray"
        }]

    if "virus" in low or "فيروسي" in cause:
        return [{
            "title_ar": "إدارة فيروسية",
            "active_ingredient": "لا يوجد مبيد علاجي مباشر",
            "trade_name": "-",
            "type": "إدارة وقائية",
            "dose": "إزالة الأجزاء المصابة + مكافحة الناقل",
            "phi": "-",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "no_direct_chemical"
        }]

    if "bacterial" in low or "بكتيري" in cause:
        return [{
            "title_ar": "الخيار 1",
            "active_ingredient": "Fixed copper",
            "trade_name": "Copper fungicide/bactericide",
            "type": "bactericide",
            "dose": "بحسب بطاقة المنتج",
            "phi": "بحسب المنتج",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "spray"
        }]

    return [
        {
            "title_ar": "الخيار 1",
            "active_ingredient": "Chlorothalonil",
            "trade_name": "Bravo",
            "type": "fungicide",
            "dose": "2 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["low", "moderate", "high"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 2",
            "active_ingredient": "Mancozeb",
            "trade_name": "Dithane M-45",
            "type": "fungicide",
            "dose": "2 جم لكل لتر ماء",
            "phi": "14 يوم",
            "severity_fit": ["low", "moderate"],
            "spray_decision": "spray"
        },
        {
            "title_ar": "الخيار 3",
            "active_ingredient": "Azoxystrobin",
            "trade_name": "Amistar",
            "type": "fungicide",
            "dose": "1 مل لكل لتر ماء",
            "phi": "7 أيام",
            "severity_fit": ["moderate", "high"],
            "spray_decision": "spray"
        }
    ]


def choose_best_pesticide_option(options: list, severity_level: str = "moderate") -> dict:
    if not options:
        return {
            "title_ar": "غير متاح",
            "active_ingredient": "-",
            "trade_name": "-",
            "type": "-",
            "dose": "-",
            "phi": "-",
            "spray_decision": "consult"
        }

    for opt in options:
        if severity_level in opt.get("severity_fit", []):
            return opt

    return options[0]


def get_spray_decision_message(spray_decision: str, severity_level: str, best_class: str) -> str:
    if spray_decision == "no_spray":
        return "لا حاجة للرش الكيميائي حاليًا لأن النبات سليم أو لا توجد إصابة واضحة."
    if spray_decision == "no_direct_chemical":
        return "لا يوجد مبيد علاجي مباشر لهذه الإصابة، ويُنصح بالاعتماد على الإدارة الوقائية ومكافحة الناقل وإزالة الأجزاء المصابة."
    if spray_decision == "consult":
        return "يُنصح بمراجعة مهندس زراعي أو مرشد زراعي قبل تطبيق أي برنامج مكافحة."
    if severity_level == "low":
        return "يمكن البدء بخيار وقائي أو مبيد خفيف مناسب مع متابعة الحالة ميدانيًا."
    if severity_level == "moderate":
        return "ينصح بالبدء في برنامج رش مناسب مع تحسين التهوية وتقليل الرطوبة ومتابعة الحالة."
    if severity_level == "high":
        return "الإصابة مرتفعة؛ يوصى بالتدخل السريع، إزالة الأجزاء الشديدة الإصابة، وتطبيق برنامج مكافحة مسجل للمحصول."

    return "اتبع توصية المختص وبطاقة المنتج المسجل."


def get_pesticide_program(best_class: str, cause: str = "", severity_level: str = "moderate") -> dict:
    key = normalize_pesticide_key(best_class, cause)

    if key in ["HEALTHY", "GENERIC"]:
        options = get_default_pesticide_program(best_class, cause, severity_level)
    else:
        options = PESTICIDE_DATABASE.get(key, get_default_pesticide_program(best_class, cause, severity_level))

    best_option = choose_best_pesticide_option(options, severity_level)

    return {
        "main": {
            "title_ar": best_option.get("title_ar", "الخيار الأساسي"),
            "active_ingredient": best_option.get("active_ingredient", "-"),
            "trade_name": best_option.get("trade_name", "-"),
            "type": best_option.get("type", "-"),
            "dose": best_option.get("dose", "-"),
            "phi": best_option.get("phi", "-"),
            "spray_decision": best_option.get("spray_decision", "consult"),
            "spray_message_ar": get_spray_decision_message(
                best_option.get("spray_decision", "consult"),
                severity_level,
                best_class
            )
        },
        "options": [
            {
                "title_ar": opt.get("title_ar", f"الخيار {i+1}"),
                "active_ingredient": opt.get("active_ingredient", "-"),
                "trade_name": opt.get("trade_name", "-"),
                "type": opt.get("type", "-"),
                "dose": opt.get("dose", "-"),
                "phi": opt.get("phi", "-"),
                "spray_decision": opt.get("spray_decision", "consult"),
                "spray_message_ar": get_spray_decision_message(
                    opt.get("spray_decision", "consult"),
                    severity_level,
                    best_class
                )
            }
            for i, opt in enumerate(options)
        ]
    }
# =========================
# الطقس + المزارعون + الشدة + GradCAM
# =========================

def get_farmers_by_region(region: str):
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT *
        FROM farmers
        WHERE region = ?
        ORDER BY id DESC
    """, (region,)).fetchall()
    conn.close()
    return rows


def send_sms_to_region_farmers(region: str, disease_name: str, risk_score: float):
    farmers = get_farmers_by_region(region)
    sent_results = []

    for farmer in farmers:
        message = (
            f"تنبيه زراعي: تم رصد خطر انتشار {disease_name} "
            f"في منطقة {region} بدرجة {round(risk_score, 2)}%."
        )

        try:
            res = sms_service.send_sms(farmer["phone"], message)
        except Exception as e:
            res = {"success": False, "error": str(e)}

        sent_results.append({
            "farmer": farmer["name"],
            "phone": farmer["phone"],
            "result": res
        })

    return sent_results


def get_live_weather(latitude: float, longitude: float) -> dict:
    query = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "auto"
    })

    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    data = fetch_json(url)
    current = data.get("current", {})

    return {
        "temperature": float(current.get("temperature_2m", 25)),
        "humidity": float(current.get("relative_humidity_2m", 70)),
        "rainfall": float(current.get("precipitation", 0)),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": data.get("timezone", "auto")
    }


def compute_real_severity(img_bgr: np.ndarray, cam_gray=None) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    yellow_mask = cv2.inRange(hsv, (12, 25, 40), (40, 255, 255))
    brown_mask1 = cv2.inRange(hsv, (0, 20, 10), (18, 255, 180))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))

    lesion_mask = cv2.bitwise_or(yellow_mask, brown_mask1)
    lesion_mask = cv2.bitwise_or(lesion_mask, dark_mask)

    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)

    if cam_gray is not None:
        cam_bin = (cam_gray > 120).astype(np.uint8) * 255
        lesion_mask = cv2.bitwise_and(lesion_mask, cam_bin)

    leaf_area = int(np.count_nonzero(leaf_mask))
    lesion_area = int(np.count_nonzero(lesion_mask))
    severity_pct = 0.0 if leaf_area == 0 else (lesion_area / leaf_area) * 100.0

    if severity_pct < 3:
        level_ar, level_en, level_code = "منخفضة", "Low", "low"
    elif severity_pct < 15:
        level_ar, level_en, level_code = "متوسطة", "Moderate", "moderate"
    else:
        level_ar, level_en, level_code = "مرتفعة", "High", "high"

    overlay = img_bgr.copy()
    overlay[lesion_mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(img_bgr, 0.72, overlay, 0.28, 0)

    return {
        "severity_percent_est": round(float(severity_pct), 2),
        "severity_level": level_code,
        "label": {"ar": level_ar, "en": level_en},
        "lesion_area_px": lesion_area,
        "leaf_area_px": leaf_area,
        "severity_overlay_b64": pil_to_base64(np_to_pil(vis))
    }


def gradcam_overlay(img_pil: Image.Image, class_idx: int):
    gradients = []
    activations = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = transform(img_pil).unsqueeze(0)

    model.zero_grad()
    out = model(x)
    score = out[0, class_idx]
    score.backward()

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam_uint8 = np.uint8(cam * 255)

    img_bgr = pil_to_cv(img_pil)
    cam_resized = cv2.resize(cam_uint8, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

    h1.remove()
    h2.remove()

    return pil_to_base64(np_to_pil(overlay)), cam_resized


def build_pdf_bytes(result: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    font_name = "ARABIC_FONT" if AR_FONT_REGISTERED else "Helvetica"
    c.setFont(font_name, 14)

    y = h - 50

    lines = [
        "تقرير تشخيص أمراض النبات",
        f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"أفضل تشخيص: {result['best_prediction']['class_name']}",
        f"نسبة الثقة: {result['best_prediction']['confidence']}%",
        f"النبات: {result['disease_info']['plant']}",
        f"المرض: {result['disease_info']['disease_ar']}",
        f"المسبب: {result['disease_info']['cause']}",
        f"شدة الإصابة: {result['severity']['label']['ar']} ({result['severity']['severity_percent_est']}%)",
        f"اسم المزارع: {result.get('farmer_name', 'غير محدد')}",
        f"اسم المزرعة: {result.get('farm_name', 'غير محدد')}",
        f"المحصول: {result.get('crop', 'غير محدد')}",
        f"المنطقة: {result.get('region', 'غير محدد')}",
        f"المدينة: {result.get('city', 'غير محدد')}",
        "",
        "التوصيات:"
    ]

    for line in lines:
        text_to_draw = fix_arabic(line) if AR_FONT_REGISTERED else line
        c.drawRightString(w - 40, y, text_to_draw)
        y -= 20

    for item in result["disease_info"]["advice"]:
        text_to_draw = fix_arabic("- " + item) if AR_FONT_REGISTERED else "- " + item
        c.drawRightString(w - 40, y, text_to_draw)
        y -= 18

    y -= 10
    c.drawRightString(w - 40, y, fix_arabic("برنامج المبيد المقترح:") if AR_FONT_REGISTERED else "برنامج المبيد المقترح:")
    y -= 20

    pesticide_lines = [
        result["pesticide_suggestion"]["title_ar"],
        f"المادة الفعالة: {result['pesticide_suggestion'].get('active_ingredient', '-')}",
        f"الاسم التجاري: {result['pesticide_suggestion'].get('trade_name', '-')}",
        f"النوع: {result['pesticide_suggestion'].get('type', '-')}",
        f"الجرعة: {result['pesticide_suggestion'].get('dose', '-')}",
        f"فترة الأمان: {result['pesticide_suggestion'].get('phi', '-')}",
        "قرار الرش:",
        result["pesticide_suggestion"].get("spray_message_ar", "-")
    ]

    for text in pesticide_lines:
        text_to_draw = fix_arabic(text) if AR_FONT_REGISTERED else text
        c.drawRightString(w - 40, y, text_to_draw)
        y -= 18 if text != "قرار الرش:" else 25

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def detect_bullseye_pattern(img_pil: Image.Image) -> bool:
    """
    كشف مبسط لنمط الحلقات المتراكزة لترجيح اللفحة المبكرة
    """
    img_bgr = pil_to_cv(img_pil)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=18,
        minRadius=6,
        maxRadius=80
    )

    return circles is not None
def infer_result_from_image(img_pil: Image.Image, lang: str = "ar") -> dict:
    x = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs_tensor = torch.softmax(out, dim=1)

    probs = probs_tensor[0].cpu().numpy()

    # أعلى 3 تنبؤات
    top_indices_all = np.argsort(probs)[::-1]
    top3_indices = top_indices_all[:3]

    # أعلى تشخيصين
    top1_index = int(top_indices_all[0])
    top2_index = int(top_indices_all[1])

    top1_name = classes[top1_index]
    top2_name = classes[top2_index]

    top1_conf = float(probs[top1_index])
    top2_conf = float(probs[top2_index])

    # كشف bullseye pattern لترجيح اللفحة المبكرة
    bullseye_detected = detect_bullseye_pattern(img_pil)

    if bullseye_detected:
        if "Early_blight" in top1_name or "Early blight" in top1_name:
            top1_conf += 0.15
        if "Early_blight" in top2_name or "Early blight" in top2_name:
            top2_conf += 0.15

    # إعادة ترتيب أعلى تشخيصين بعد الترجيح
    if top2_conf > top1_conf:
        top1_name, top2_name = top2_name, top1_name
        top1_conf, top2_conf = top2_conf, top1_conf
        top1_index, top2_index = top2_index, top1_index

    gap = abs(top1_conf - top2_conf)

    # مستوى الثقة
    if top1_conf >= 0.80:
        confidence_level = "عالية"
    elif top1_conf >= 0.60:
        confidence_level = "متوسطة"
    else:
        confidence_level = "منخفضة"

    # حالة القرار
    if top1_conf < 0.60:
        decision_status = "غير مؤكد"
        decision_text = f"الثقة منخفضة ويوجد تشابه بين {top1_name} و {top2_name}"
        similar_case = True
    elif gap < 0.15:
        decision_status = "تشابه بين مرضين"
        decision_text = f"يوجد تشابه بين {top1_name} و {top2_name}"
        similar_case = True
    else:
        decision_status = "تشخيص محتمل"
        decision_text = f"التشخيص الأقرب هو {top1_name}"
        similar_case = False

    # التشخيص الأساسي النهائي
    best_idx = top1_index
    best_class = top1_name
    best_conf = top1_conf

    # أفضل 3 تنبؤات
    top3 = []
    top_3_predictions = []
    for idx in top3_indices:
        cls = classes[int(idx)]
        conf = float(probs[int(idx)])

        # لو كان هذا هو Early blight وتم كشف bullseye نرفع النسبة المعروضة
        if cls == best_class:
            conf = best_conf

        top3.append({
            "class": cls,
            "confidence": conf
        })

        top_3_predictions.append({
            "class_name": cls,
            "confidence": round(conf * 100, 2)
        })

    info = DISEASE_INFO.get(best_class, {
        "plant": "غير معروف",
        "disease_ar": best_class,
        "cause": "غير معروف",
        "advice": ["لا توجد معلومات مضافة لهذه الفئة بعد."]
    })

    gradcam_b64, cam_gray = gradcam_overlay(img_pil, best_idx)
    severity = compute_real_severity(pil_to_cv(img_pil), cam_gray)

    pesticide_program = get_pesticide_program(
        best_class=best_class,
        cause=info.get("cause", ""),
        severity_level=severity.get("severity_level", "moderate")
    )

    return {
        "success": True,
        "pred_class": best_class,
        "pred_label": info.get("disease_ar", best_class),
        "confidence": best_conf,
        "recommendations": info.get("advice", []),
        "severity": severity,
        "gradcam_overlay_b64": gradcam_b64,

        "best_prediction": {
            "class_name": best_class,
            "confidence": round(best_conf * 100, 2)
        },

        "top3": top3,
        "top_3_predictions": top_3_predictions,

        "disease_info": {
            "plant": info.get("plant", "غير معروف"),
            "disease_ar": info.get("disease_ar", best_class),
            "cause": info.get("cause", "غير معروف"),
            "advice": info.get("advice", [])
        },

        "pesticide_suggestion": pesticide_program["main"],
        "pesticide_options": pesticide_program["options"],

        # الحقول الجديدة للواجهة
        "top1_disease": top1_name,
        "top1_confidence": round(top1_conf, 4),
        "top2_disease": top2_name,
        "top2_confidence": round(top2_conf, 4),
        "confidence_level": confidence_level,
        "decision_status": decision_status,
        "decision_text": decision_text,
        "similar_case": similar_case,
        "bullseye_detected": bullseye_detected
    }
    
# =========================
# التوقع والمخاطر
# =========================

def calc_weather_risk(temperature, humidity, rainfall):
    score = 0

    if humidity >= 85:
        score += 40
    elif humidity >= 70:
        score += 25
    else:
        score += 10

    if 18 <= temperature <= 28:
        score += 30
    else:
        score += 10

    if rainfall >= 10:
        score += 30
    elif rainfall >= 2:
        score += 15
    else:
        score += 5

    return min(score, 100)


def calc_history_risk(region="غير محدد", disease=""):
    conn = get_db_connection()
    cur = conn.cursor()

    if disease and disease != "الكل":
        row = cur.execute("""
            SELECT COUNT(*) as cnt, AVG(severity_percent) as avg_severity
            FROM diagnoses
            WHERE region = ? AND disease_ar = ?
        """, (region, disease)).fetchone()
    else:
        row = cur.execute("""
            SELECT COUNT(*) as cnt, AVG(severity_percent) as avg_severity
            FROM diagnoses
            WHERE region = ?
        """, (region,)).fetchone()

    conn.close()

    cases = row["cnt"] if row and row["cnt"] else 0
    avg = row["avg_severity"] if row and row["avg_severity"] else 0

    score = 0
    if cases > 10:
        score += 25
    elif cases > 5:
        score += 15
    elif cases > 1:
        score += 8

    if avg > 20:
        score += 20
    elif avg > 10:
        score += 10

    return {
        "cases_count": cases,
        "avg_severity": round(float(avg), 2) if avg else 0,
        "history_score": score
    }


def risk_level(score):
    if score < 35:
        return {"level_ar": "منخفض", "color": "green"}
    elif score < 65:
        return {"level_ar": "متوسط", "color": "orange"}
    else:
        return {"level_ar": "مرتفع", "color": "red"}


def build_alert_message(region: str, city: str, disease: str, level_ar: str, score: float) -> str:
    return f"تنبيه {level_ar}: خطر انتشار المرض '{disease}' في المنطقة '{region}' والمدينة '{city}' بدرجة {round(score, 2)}%."


def build_alert_recommendation(level_ar: str) -> str:
    if level_ar == "مرتفع":
        return "تكثيف الفحص الحقلي اليومي، وخفض الرطوبة، والبدء بإجراءات وقائية فورية."
    elif level_ar == "متوسط":
        return "الفحص كل 2-3 أيام، ومراقبة الرطوبة، والاستعداد للإجراءات الوقائية."
    return "الاستمرار في الفحص الدوري والنظافة الزراعية."


def should_save_alert(region: str, city: str, disease_name: str, risk_level_ar: str) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()

    row = cur.execute("""
        SELECT created_at
        FROM alerts
        WHERE region = ? AND city = ? AND disease_name = ? AND risk_level = ?
        ORDER BY id DESC
        LIMIT 1
    """, (region, city, disease_name, risk_level_ar)).fetchone()

    conn.close()

    if not row:
        return True

    try:
        created_at = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
        return datetime.now() - created_at > timedelta(hours=1)
    except Exception:
        return True


def evaluate_live_risk(region, city, disease, crop, latitude=None, longitude=None, save_if_high=True):
    lat, lon, source = resolve_region_or_city_coords(
        region=region,
        city=city,
        latitude=latitude,
        longitude=longitude
    )

    live_weather = get_live_weather(lat, lon)
    weather_score = calc_weather_risk(
        live_weather["temperature"],
        live_weather["humidity"],
        live_weather["rainfall"]
    )

    history = calc_history_risk(region, disease)
    final_score = min(weather_score + history["history_score"], 100)
    level = risk_level(final_score)

    if level["level_ar"] == "مرتفع":
        recommendations = [
            "تكثيف الفحص اليومي للمزرعة.",
            "بدء برنامج وقائي أو استجابة فورية حسب المرض والمحصول.",
            "خفض الرطوبة وتحسين التهوية.",
            "عزل أو إزالة الأجزاء شديدة الإصابة."
        ]
    elif level["level_ar"] == "متوسط":
        recommendations = [
            "الفحص الدوري كل 2-3 أيام.",
            "تحسين إدارة الري وتقليل البلل الورقي.",
            "الاستعداد لبرنامج وقائي إذا ظهرت أعراض أولية."
        ]
    else:
        recommendations = [
            "استمرار المراقبة الروتينية.",
            "الحفاظ على النظافة الزراعية.",
            "لا يوجد خطر مرتفع حاليًا."
        ]

    disease_name = disease if disease != "الكل" else "خطر مرض عام"

    if save_if_high and level["level_ar"] in ["متوسط", "مرتفع"]:
        if should_save_alert(region, city, disease_name, level["level_ar"]):
            save_alert(
                region=region,
                city=city,
                disease_name=disease_name,
                crop=crop,
                risk_score=final_score,
                risk_level=level["level_ar"],
                alert_message=build_alert_message(region, city, disease_name, level["level_ar"], final_score),
                recommendation=build_alert_recommendation(level["level_ar"])
            )

            if level["level_ar"] == "مرتفع":
                try:
                    send_sms_to_region_farmers(region=region, disease_name=disease_name, risk_score=final_score)
                except Exception:
                    pass

    return {
        "region": region,
        "city": city,
        "crop": crop,
        "disease": disease,
        "coords": {
            "latitude": lat,
            "longitude": lon,
            "source": source
        },
        "live_weather": live_weather,
        "weather_score": round(weather_score, 2),
        "history": history,
        "final_score": round(final_score, 2),
        "risk_level": level,
        "recommendations": recommendations
    }


# =========================
# المجدول الوطني
# =========================

scheduler = BackgroundScheduler()

def national_risk_scan():
    for region, info in REGION_COORDS.items():
        try:
            evaluate_live_risk(
                region=region,
                city=info["city"],
                disease="الكل",
                crop="غير محدد",
                latitude=info["lat"],
                longitude=info["lon"],
                save_if_high=True
            )
        except Exception:
            pass

if not scheduler.running:
    scheduler.add_job(
        national_risk_scan,
        "interval",
        minutes=60,
        id="national_risk_scan_job",
        replace_existing=True
    )
    scheduler.start()


# =========================
# Root / Health
# =========================


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "num_classes": num_classes
    }


# =========================
# Predict / Report
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = Query("ar"),
    city: str = Query("غير محدد"),
    region: str = Query("غير محدد"),
    farmer_name: str = Query("غير محدد"),
    farm_name: str = Query("غير محدد"),
    crop: str = Query("غير محدد"),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None)
):
    try:
        image_bytes = await file.read()
        image_path = save_upload_file(image_bytes, file.filename or "upload.jpg")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = extract_disease_region(img)
        result = infer_result_from_image(img, lang=lang)
        lat, lon, _ = resolve_region_or_city_coords(region=region, city=city, latitude=latitude, longitude=longitude)

        result["farmer_name"] = farmer_name
        result["farm_name"] = farm_name
        result["crop"] = crop
        result["city"] = city
        result["region"] = region
        result["latitude"] = lat
        result["longitude"] = lon
        result["image_path"] = image_path

        save_diagnosis(
            farmer_name=farmer_name,
            farm_name=farm_name,
            crop=crop,
            plant=result["disease_info"]["plant"],
            disease_class=result["pred_class"],
            disease_ar=result["disease_info"]["disease_ar"],
            confidence=result["confidence"],
            severity_percent=result["severity"]["severity_percent_est"],
            cause=result["disease_info"]["cause"],
            city=city,
            region=region,
            latitude=lat,
            longitude=lon,
            image_path=image_path
        )

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...), lang: str = Query("ar")):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = extract_disease_region(img)
        result = infer_result_from_image(img, lang=lang)

        return JSONResponse({
            "success": True,
            "best_prediction": result["best_prediction"],
            "disease_info": result["disease_info"],
            "severity": result["severity"],
            "pesticide_suggestion": result["pesticide_suggestion"],
            "pesticide_options": result["pesticide_options"]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    lang: str = Query("ar"),
    city: str = Query("غير محدد"),
    region: str = Query("غير محدد"),
    farmer_name: str = Query("غير محدد"),
    farm_name: str = Query("غير محدد"),
    crop: str = Query("غير محدد")
):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = extract_disease_region(img)
        result = infer_result_from_image(img, lang=lang)

        result["farmer_name"] = farmer_name
        result["farm_name"] = farm_name
        result["crop"] = crop
        result["city"] = city
        result["region"] = region

        pdf_bytes = build_pdf_bytes(result)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=phytologic_report.pdf"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# =========================
# Stats
# =========================

@app.get("/stats/summary")
def stats_summary():
    conn = get_db_connection()
    cur = conn.cursor()

    total_cases = cur.execute("SELECT COUNT(*) AS total FROM diagnoses").fetchone()["total"]
    avg_severity = cur.execute("SELECT AVG(severity_percent) AS avg_severity FROM diagnoses").fetchone()["avg_severity"]

    top_disease_row = cur.execute("""
        SELECT disease_ar, COUNT(*) AS count
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY count DESC
        LIMIT 1
    """).fetchone()

    top_region_row = cur.execute("""
        SELECT region, COUNT(*) AS count
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    return {
        "total_cases": total_cases or 0,
        "avg_severity": round(float(avg_severity), 2) if avg_severity is not None else 0,
        "top_disease": {
            "name": top_disease_row["disease_ar"],
            "count": top_disease_row["count"]
        } if top_disease_row else None,
        "top_region": {
            "name": top_region_row["region"],
            "count": top_region_row["count"]
        } if top_region_row else None
    }


@app.get("/stats/diseases")
def stats_diseases():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT disease_ar, COUNT(*) AS count
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [{"disease": row["disease_ar"], "count": row["count"]} for row in rows]


@app.get("/stats/regions")
def stats_regions():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT region, COUNT(*) AS count
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [{"region": row["region"], "count": row["count"]} for row in rows]


@app.get("/stats/severity")
def stats_severity():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT disease_ar, AVG(severity_percent) AS avg_severity
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY avg_severity DESC
    """).fetchall()
    conn.close()

    return [
        {
            "disease": row["disease_ar"],
            "avg_severity": round(float(row["avg_severity"]), 2) if row["avg_severity"] is not None else 0
        }
        for row in rows
    ]


@app.get("/stats/all")
def stats_all():
    conn = get_db_connection()
    cur = conn.cursor()

    total_cases = cur.execute("SELECT COUNT(*) AS total FROM diagnoses").fetchone()["total"]
    avg_severity = cur.execute("SELECT AVG(severity_percent) AS avg_severity FROM diagnoses").fetchone()["avg_severity"]

    diseases = cur.execute("""
        SELECT disease_ar, COUNT(*) AS count
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY count DESC
    """).fetchall()

    regions = cur.execute("""
        SELECT region, COUNT(*) AS count
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()

    severities = cur.execute("""
        SELECT disease_ar, AVG(severity_percent) AS avg_severity
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY avg_severity DESC
    """).fetchall()

    conn.close()

    return {
        "summary": {
            "total_cases": total_cases or 0,
            "avg_severity": round(float(avg_severity), 2) if avg_severity is not None else 0
        },
        "diseases": [{"disease": row["disease_ar"], "count": row["count"]} for row in diseases],
        "regions": [{"region": row["region"], "count": row["count"]} for row in regions],
        "severity": [
            {
                "disease": row["disease_ar"],
                "avg_severity": round(float(row["avg_severity"]), 2) if row["avg_severity"] is not None else 0
            }
            for row in severities
        ]
    }


# =========================
# Forecast
# =========================

@app.get("/weather/current")
def weather_current(
    region: str = Query("غير محدد"),
    city: str = Query("غير محدد"),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None)
):
    try:
        lat, lon, source = resolve_region_or_city_coords(region=region, city=city, latitude=latitude, longitude=longitude)
        live_weather = get_live_weather(lat, lon)
        return {
            "success": True,
            "region": region,
            "city": city,
            "coords": {
                "latitude": lat,
                "longitude": lon,
                "source": source
            },
            "weather": live_weather
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/forecast/risk")
def forecast_risk(
    region: str = Query("غير محدد"),
    city: str = Query("غير محدد"),
    disease: str = Query("الكل"),
    crop: str = Query("غير محدد"),
    temperature: float = Query(25),
    humidity: float = Query(70),
    rainfall: float = Query(0),
    save_if_high: bool = Query(True)
):
    weather_score = calc_weather_risk(temperature, humidity, rainfall)
    history = calc_history_risk(region, disease)
    final_score = min(weather_score + history["history_score"], 100)
    level = risk_level(final_score)

    recommendations = [
        "مراقبة الحقل بشكل دوري",
        "تقليل الرطوبة الحرة على الأوراق",
        "إزالة الأوراق المصابة مبكرًا",
        "تطبيق برنامج وقائي عند ارتفاع الخطر"
    ]

    if save_if_high and level["level_ar"] in ["متوسط", "مرتفع"]:
        disease_name = disease if disease != "الكل" else "خطر مرض عام"
        if should_save_alert(region, city, disease_name, level["level_ar"]):
            save_alert(
                region=region,
                city=city,
                disease_name=disease_name,
                crop=crop,
                risk_score=final_score,
                risk_level=level["level_ar"],
                alert_message=build_alert_message(region, city, disease_name, level["level_ar"], final_score),
                recommendation=build_alert_recommendation(level["level_ar"])
            )

    return {
        "region": region,
        "city": city,
        "crop": crop,
        "disease": disease,
        "weather_score": weather_score,
        "history": history,
        "final_score": final_score,
        "risk_level": level,
        "recommendations": recommendations
    }


@app.get("/forecast/live-risk")
def forecast_live_risk(
    region: str = Query("غير محدد"),
    city: str = Query("غير محدد"),
    disease: str = Query("الكل"),
    crop: str = Query("غير محدد"),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
    save_if_high: bool = Query(True)
):
    try:
        return evaluate_live_risk(
            region=region,
            city=city,
            disease=disease,
            crop=crop,
            latitude=latitude,
            longitude=longitude,
            save_if_high=save_if_high
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/forecast/live-regions")
def forecast_live_regions(
    disease: str = Query("الكل"),
    crop: str = Query("غير محدد"),
    save_if_high: bool = Query(False)
):
    results = []

    for region, info in REGION_COORDS.items():
        try:
            result = evaluate_live_risk(
                region=region,
                city=info["city"],
                disease=disease,
                crop=crop,
                latitude=info["lat"],
                longitude=info["lon"],
                save_if_high=save_if_high
            )
            results.append({
                "region": region,
                "city": info["city"],
                "final_score": result["final_score"],
                "risk_level_ar": result["risk_level"]["level_ar"],
                "color": result["risk_level"]["color"],
                "weather": result["live_weather"],
                "cases_count": result["history"]["cases_count"],
                "avg_severity": result["history"]["avg_severity"]
            })
        except Exception as e:
            results.append({
                "region": region,
                "city": info["city"],
                "error": str(e),
                "final_score": 0,
                "risk_level_ar": "غير متاح",
                "color": "gray",
                "weather": {"temperature": None, "humidity": None, "rainfall": None},
                "cases_count": 0,
                "avg_severity": 0
            })

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results


@app.get("/forecast/ai")
def forecast_ai(
    temperature: float = Query(...),
    humidity: float = Query(...),
    rainfall: float = Query(...),
    cases_count: int = Query(...),
    severity_avg: float = Query(...)
):
    try:
        result = forecast_ai_service.predict_cases(
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            cases_count=cases_count,
            severity_avg=severity_avg
        )

        return {
            "success": True,
            "inputs": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "cases_count": cases_count,
                "severity_avg": severity_avg
            },
            "forecast": result
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# =========================
# Farmers / Alerts / Map / Export
# =========================

@app.post("/farmers/register")
def register_farmer(
    name: str = Query(...),
    phone: str = Query(...),
    farm_name: str = Query(...),
    region: str = Query(...),
    city: str = Query(...),
    crop: str = Query(...)
):
    try:
        save_farmer(name=name, phone=phone, farm_name=farm_name, region=region, city=city, crop=crop)
        return {"success": True, "message": "Farmer registered successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/farmers")
def get_farmers(region: str = Query(None)):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if region:
            rows = cur.execute("SELECT * FROM farmers WHERE region = ? ORDER BY id DESC", (region,)).fetchall()
        else:
            rows = cur.execute("SELECT * FROM farmers ORDER BY id DESC").fetchall()

        conn.close()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "phone": row["phone"],
                "farm_name": row["farm_name"],
                "region": row["region"],
                "city": row["city"],
                "crop": row["crop"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/alerts/send-sms")
def send_sms_alert(
    region: str = Query(...),
    disease_name: str = Query(...),
    risk_score: float = Query(...)
):
    try:
        sent_results = send_sms_to_region_farmers(region=region, disease_name=disease_name, risk_score=risk_score)
        return {"success": True, "sent_count": len(sent_results), "details": sent_results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/alerts")
def get_alerts(limit: int = Query(100)):
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "region": row["region"],
            "city": row["city"],
            "disease_name": row["disease_name"],
            "crop": row["crop"],
            "risk_score": row["risk_score"],
            "risk_level": row["risk_level"],
            "alert_message": row["alert_message"],
            "recommendation": row["recommendation"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


@app.get("/alerts/summary")
def alerts_summary():
    conn = get_db_connection()
    cur = conn.cursor()

    total_alerts = cur.execute("SELECT COUNT(*) AS total FROM alerts").fetchone()["total"]
    high_alerts = cur.execute("SELECT COUNT(*) AS total FROM alerts WHERE risk_level = 'مرتفع'").fetchone()["total"]
    moderate_alerts = cur.execute("SELECT COUNT(*) AS total FROM alerts WHERE risk_level = 'متوسط'").fetchone()["total"]

    latest_alert = cur.execute("""
        SELECT *
        FROM alerts
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    return {
        "total_alerts": total_alerts or 0,
        "high_alerts": high_alerts or 0,
        "moderate_alerts": moderate_alerts or 0,
        "latest_alert": {
            "region": latest_alert["region"],
            "city": latest_alert["city"],
            "disease_name": latest_alert["disease_name"],
            "risk_score": latest_alert["risk_score"],
            "risk_level": latest_alert["risk_level"],
            "created_at": latest_alert["created_at"]
        } if latest_alert else None
    }


@app.get("/alerts/regions")
def alerts_regions():
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT region, COUNT(*) AS count
        FROM alerts
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()
    conn.close()

    return [{"region": row["region"], "count": row["count"]} for row in rows]


@app.get("/map/data")
def map_data():
    conn = get_db_connection()
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT *
        FROM diagnoses
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    result = []
    for row in rows:
        row = dict(row)

        lat = row.get("latitude")
        lon = row.get("longitude")
        region = row.get("region") or "غير محدد"

        if (lat is None or lon is None) and region in REGION_COORDS:
            lat = REGION_COORDS[region]["lat"]
            lon = REGION_COORDS[region]["lon"]

        result.append({
            "id": row.get("id"),
            "farmer_name": row.get("farmer_name", "غير محدد"),
            "farm_name": row.get("farm_name", "غير محدد"),
            "crop": row.get("crop", "غير محدد"),
            "plant": row.get("plant", "غير محدد"),
            "disease": row.get("disease_ar", "غير محدد"),
            "confidence": round(float(row.get("confidence", 0) or 0), 4),
            "severity": round(float(row.get("severity_percent", 0) or 0), 2),
            "cause": row.get("cause", "غير محدد"),
            "city": row.get("city", "غير محدد"),
            "region": region,
            "latitude": lat,
            "longitude": lon,
            "image_path": row.get("image_path", ""),
            "created_at": row.get("created_at", "")
        })

    return result


@app.get("/map/summary")
def map_summary():
    conn = get_db_connection()
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT region, COUNT(*) AS count, AVG(severity_percent) AS avg_severity
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()

    conn.close()

    result = []
    for row in rows:
        region_name = row["region"] or "غير محدد"
        coords = REGION_COORDS.get(region_name)

        result.append({
            "region": region_name,
            "count": row["count"],
            "avg_severity": round(float(row["avg_severity"]), 2) if row["avg_severity"] is not None else 0,
            "latitude": coords["lat"] if coords else None,
            "longitude": coords["lon"] if coords else None
        })

    return result


@app.get("/export/diagnoses/csv")
def export_diagnoses_csv():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM diagnoses").fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')

    if rows:
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow([row[k] for k in row.keys()])

    csv_text = output.getvalue()
    csv_bytes = csv_text.encode("utf-8-sig")

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=diagnoses.csv"}
    )


@app.get("/export/diagnoses/json")
def export_diagnoses_json():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM diagnoses").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/export/alerts/csv")
def export_alerts_csv():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM alerts").fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')

    if rows:
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow([row[k] for k in row.keys()])

    csv_text = output.getvalue()
    csv_bytes = csv_text.encode("utf-8-sig")

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=alerts.csv"}
    )


@app.get("/export/alerts/json")
def export_alerts_json():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM alerts").fetchall()
    conn.close()
    return [dict(r) for r in rows]
    @app.get("/{page_name}")
def open_page_legacy(page_name: str):
    blocked_prefixes = {
        "predict", "predict-frame", "report",
        "stats", "forecast", "weather",
        "farmers", "alerts", "map", "export",
        "health", "static", "pages"
    }

    first_segment = page_name.split("/")[0]
    if first_segment in blocked_prefixes:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    file_path = page_name if page_name.endswith(".html") else f"{page_name}.html"

    if os.path.exists(file_path):
        return FileResponse(file_path)

    return JSONResponse({"error": f"{file_path} not found"}, status_code=404)
