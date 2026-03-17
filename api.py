# =========================
# IMPORTS
# =========================

import io
import os
import uuid
import json
import base64
import urllib.parse
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from torchvision import transforms, models

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import arabic_reshaper
from bidi.algorithm import get_display

from database import init_db, save_diagnosis, save_alert, save_farmer, get_connection
from ai_forecast_service import AIForecastService
from sms_service import SMSService

try:
    from disease_info import DISEASE_INFO
except Exception:
    DISEASE_INFO = None

try:
    from disease_info import get_disease_info_by_class as imported_get_disease_info_by_class
except Exception:
    imported_get_disease_info_by_class = None


# =========================
# DATABASE CONNECTION
# =========================

def get_db_connection():
    return get_connection()


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Phytologic AI Platform",
    description="Plant Disease Diagnosis and Smart Agriculture System",
    version="4.1"
)


# =========================
# PATHS / STORAGE
# =========================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", ".")
STATIC_DIR = os.path.join(BASE_PATH, "static")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
FONTS_DIR = os.path.join(BASE_PATH, "fonts")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)


# =========================
# CORS
# =========================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != [""] else ["*"],
    allow_credentials=False if ALLOWED_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# STATIC FILES
# =========================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================
# PAGE ROUTING
# =========================

PAGE_FILE_MAP = {
    "index_home": "index_home.html",
    "dashboard": "dashboard.html",
    "alerts": "alerts.html",
    "report_center": "report_center.html",
    "farmers": "farmers.html",
    "admin": "admin.html",
    "national_center": "national_center.html",
    "ops": "ops.html",
    "forecast": "forecast.html",
    "forecast_ai": "forecast_ai.html",
    "layout_shell": "layout_shell_pro.html",
    "layout_shell_ultra": "layout_shell_ultra.html",
    "login": "login.html",
    "register": "register.html",
}

OPTIONAL_PAGE_CANDIDATES = {
    "layout_shell_pro": "layout_shell_pro.html",
    "layout_shell_ultra": "layout_shell_ultra.html",
}


def get_index_file_path():
    index_candidates = [
        os.path.join(BASE_PATH, "index.html"),
        os.path.join(BASE_PATH, "index_home.html"),
    ]
    for path in index_candidates:
        if os.path.exists(path):
            return path
    return None


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    index_path = get_index_file_path()

    if index_path:
        return FileResponse(index_path)

    if "HEAD" in []:
        return Response(status_code=404)

    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.head("/")
def root_head():
    index_path = get_index_file_path()
    if index_path:
        return Response(status_code=200)
    return Response(status_code=404)


@app.get("/pages/{page_name}")
def open_page(page_name: str):
    safe_page = (page_name or "").strip().lower()

    filename = PAGE_FILE_MAP.get(safe_page) or OPTIONAL_PAGE_CANDIDATES.get(safe_page)
    if not filename:
        return JSONResponse({"error": "page not allowed"}, status_code=404)

    file_path = os.path.join(BASE_PATH, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)

    return JSONResponse({"error": "page file not found"}, status_code=404)


# =========================
# INIT DB / SERVICES
# =========================

init_db()

forecast_ai_service = AIForecastService()

sms_service = SMSService(
    app_sid=os.getenv("UNIFONIC_APP_SID", ""),
    sender=os.getenv("SMS_SENDER", "Phytologic")
)


# =========================
# MODEL FILES
# =========================

MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", os.path.join(DATA_DIR, "plant_disease_model_v3.pth"))
IMG_SIZE = 160


def download_file_if_missing(local_path: str, url: str):
    if os.path.exists(local_path):
        return
    urllib.request.urlretrieve(url, local_path)


download_file_if_missing(
    MODEL_PATH,
    "https://github.com/aboremasx0x-boop/phytologic-ai-platform/releases/download/v1/plant_disease_model_v3.pth"
)

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
# ARABIC FONTS FOR PDF
# =========================

def ensure_font_file(local_path: str, download_url: str):
    if os.path.exists(local_path):
        return True
    try:
        urllib.request.urlretrieve(download_url, local_path)
        print(f"Downloaded font successfully: {local_path}")
        return True
    except Exception as e:
        print(f"Failed to download font {local_path}: {e}")
        return False


# الأفضل ترفع الخط يدويًا داخل المشروع:
# fonts/Amiri-Regular.ttf
# fonts/NotoNaskhArabic-Regular.ttf
#
# وإذا لم يوجد، نحاول تنزيل Noto فقط لأن رابطه أكثر استقرارًا.
NOTO_FONT_PATH = os.path.join(FONTS_DIR, "NotoNaskhArabic-Regular.ttf")
AMIRI_FONT_PATH = os.path.join(FONTS_DIR, "Amiri-Regular.ttf")

if not os.path.exists(NOTO_FONT_PATH):
    ensure_font_file(
        NOTO_FONT_PATH,
        "https://github.com/notofonts/arabic/raw/main/fonts/ttf/NotoNaskhArabic/NotoNaskhArabic-Regular.ttf"
    )

AR_FONT_REGISTERED = False
AR_FONT_PATH_USED = None

for font_path in [
    AMIRI_FONT_PATH,                       # محلي إن وجد
    NOTO_FONT_PATH,                        # محلي أو منزّل
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf"
]:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("ARABIC_FONT", font_path))
            AR_FONT_REGISTERED = True
            AR_FONT_PATH_USED = font_path
            print(f"Arabic font registered: {font_path}")
            break
        except Exception as e:
            print("Font registration failed:", font_path, e)

print("AR_FONT_REGISTERED =", AR_FONT_REGISTERED)
print("AR_FONT_PATH_USED =", AR_FONT_PATH_USED)


# =========================
# COORDINATES
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
# HELPERS
# =========================

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp"
}


def validate_uploaded_image(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported file type. Allowed: JPEG, PNG, WEBP, BMP")


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
    ext = os.path.splitext(original_filename or "")[1].lower().strip()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path


def fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "PhytologicAI/1.0"})
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


def get_disease_info_by_class(best_class: str):
    if callable(imported_get_disease_info_by_class):
        try:
            return imported_get_disease_info_by_class(best_class)
        except Exception:
            pass

    if isinstance(DISEASE_INFO, dict):
        return DISEASE_INFO.get(
            best_class,
            {
                "plant": "غير معروف",
                "disease_ar": "غير معروف",
                "cause": "غير معروف",
                "advice": []
            }
        )

    return {
        "plant": "غير معروف",
        "disease_ar": "غير معروف",
        "cause": "غير معروف",
        "advice": []
    }


# =========================
# PESTICIDE DATABASE
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
        }
    ]
}


def normalize_pesticide_key(best_class: str, cause: str = "") -> str:
    name = (best_class or "").replace("-", "_").replace(" ", "_")
    low = name.lower()
    cause_low = (cause or "").lower()

    if "healthy" in low or "سليم" in cause:
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

    if "healthy" in low or "سليم" in (cause or ""):
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

    return [{
        "title_ar": "الخيار 1",
        "active_ingredient": "Chlorothalonil",
        "trade_name": "Bravo",
        "type": "fungicide",
        "dose": "2 مل لكل لتر ماء",
        "phi": "7 أيام",
        "severity_fit": ["low", "moderate", "high"],
        "spray_decision": "spray"
    }]


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
        return "لا يوجد مبيد علاجي مباشر لهذه الإصابة، ويُنصح بالإدارة الوقائية ومكافحة الناقل."
    if spray_decision == "consult":
        return "يُنصح بمراجعة مختص قبل تطبيق أي برنامج مكافحة."
    if severity_level == "low":
        return "يمكن البدء بخيار وقائي أو مبيد خفيف مناسب مع المتابعة."
    if severity_level == "moderate":
        return "ينصح بالبدء في برنامج رش مناسب مع تحسين التهوية وتقليل الرطوبة."
    if severity_level == "high":
        return "الإصابة مرتفعة؛ يوصى بالتدخل السريع وإزالة الأجزاء الشديدة الإصابة."
    return "اتبع توصية المختص وبطاقة المنتج."


def get_pesticide_program(best_class: str, cause: str = "", severity_level: str = "moderate") -> dict:
    key = normalize_pesticide_key(best_class, cause)

    if key in ["HEALTHY", "GENERIC"]:
        options = get_default_pesticide_program(best_class, cause, severity_level)
    else:
        options = PESTICIDE_DATABASE.get(
            key,
            get_default_pesticide_program(best_class, cause, severity_level)
        )

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
        "options": options
    }


# =========================
# WEATHER / FARMERS / SMS
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
        "longitude": longitude
    }


# =========================
# IMAGE ANALYSIS
# =========================

def compute_real_severity(img_bgr: np.ndarray, cam_gray=None) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, leaf_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    yellow_mask = cv2.inRange(hsv, (12, 25, 40), (40, 255, 255))
    brown_mask = cv2.inRange(hsv, (0, 20, 10), (18, 255, 180))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))

    lesion_mask = cv2.bitwise_or(yellow_mask, brown_mask)
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


# =========================
# QUESTIONS / RESULT BUILDERS
# =========================

def generate_diagnostic_questions(best_class: str, second_class: str):
    questions = []

    best = best_class.lower()
    second = second_class.lower()

    if "early" in best and "blight" in best:
        questions.append({
            "question": "هل تظهر بقع دائرية بها حلقات متداخلة (Bullseye)؟",
            "symptom": "bullseye"
        })
        questions.append({
            "question": "هل الأوراق السفلية هي الأكثر إصابة؟",
            "symptom": "lower_leaves"
        })

    if "septoria" in second:
        questions.append({
            "question": "هل البقع صغيرة ولها مركز رمادي؟",
            "symptom": "septoria_spots"
        })

    if "powdery" in best:
        questions.append({
            "question": "هل يوجد مسحوق أبيض على سطح الورقة؟",
            "symptom": "white_powder"
        })

    if "mildew" in best:
        questions.append({
            "question": "هل توجد طبقة بيضاء أو رمادية على الأوراق؟",
            "symptom": "mildew_layer"
        })

    return questions


def build_diagnosis_result(
    best_class,
    best_conf,
    second_class,
    second_conf,
    disease_info,
    severity,
    pesticide_program,
    weather_data,
    questions
):
    confidence_level = "منخفضة"
    if best_conf > 80:
        confidence_level = "مرتفعة"
    elif best_conf > 50:
        confidence_level = "متوسطة"

    decision_status = "مؤكد"
    if abs(best_conf - second_conf) < 5:
        decision_status = "غير مؤكد"

    return {
        "best_prediction": {
            "class_name": best_class,
            "confidence": round(float(best_conf), 2)
        },
        "second_prediction": {
            "class_name": second_class,
            "confidence": round(float(second_conf), 2)
        },
        "confidence_level": confidence_level,
        "decision_status": decision_status,
        "disease_info": disease_info,
        "severity": severity,
        "pesticide_program": pesticide_program,
        "weather": weather_data,
        "questions": questions
    }


# =========================
# PDF
# =========================

def build_pdf_bytes(result: dict):
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

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.read()


# =========================
# DIAGNOSE ENDPOINTS
# =========================

@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    farmer_name: str = "",
    farm_name: str = "",
    crop: str = "",
    city: str = "",
    region: str = "",
    latitude: float = 0,
    longitude: float = 0
):
    try:
        validate_uploaded_image(file)
        contents = await file.read()

        try:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        image_path = save_upload_file(contents, file.filename or "upload.jpg")
        img_bgr = pil_to_cv(img_pil)

        x = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]

        probs_np = probs.cpu().numpy()
        sorted_idx = np.argsort(probs_np)[::-1]

        best_idx = int(sorted_idx[0])
        second_idx = int(sorted_idx[1])

        best_class = classes[best_idx]
        second_class = classes[second_idx]

        best_conf = float(probs_np[best_idx] * 100)
        second_conf = float(probs_np[second_idx] * 100)

        gradcam_b64, cam_gray = gradcam_overlay(img_pil, best_idx)

        severity = compute_real_severity(img_bgr, cam_gray)
        severity_level = severity["severity_level"]

        disease_info = get_disease_info_by_class(best_class)

        pesticide_program = get_pesticide_program(
            best_class,
            disease_info.get("cause", ""),
            severity_level
        )

        lat, lon, weather_source = resolve_region_or_city_coords(
            region=region,
            city=city,
            latitude=latitude if latitude else None,
            longitude=longitude if longitude else None
        )

        try:
            weather_data = get_live_weather(lat, lon)
            weather_data["source"] = weather_source
        except Exception:
            weather_data = {
                "temperature": None,
                "humidity": None,
                "rainfall": None,
                "latitude": lat,
                "longitude": lon,
                "source": weather_source
            }

        questions = generate_diagnostic_questions(best_class, second_class)

        result = build_diagnosis_result(
            best_class,
            best_conf,
            second_class,
            second_conf,
            disease_info,
            severity,
            pesticide_program,
            weather_data,
            questions
        )

        result["gradcam_image"] = gradcam_b64
        result["image_path"] = image_path

        try:
            save_diagnosis(
                farmer_name=farmer_name,
                farm_name=farm_name,
                crop=crop,
                plant=disease_info.get("plant", ""),
                disease_class=best_class,
                disease_ar=disease_info.get("disease_ar", ""),
                confidence=best_conf,
                severity_percent=severity["severity_percent_est"],
                cause=disease_info.get("cause", ""),
                city=city,
                region=region,
                latitude=lat,
                longitude=lon,
                image_path=image_path
            )
        except Exception as db_error:
            print("DB save error:", db_error)

        return JSONResponse(result)

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/diagnose/pdf")
async def diagnose_and_pdf(
    file: UploadFile = File(...),
    farmer_name: str = "",
    farm_name: str = "",
    crop: str = "",
    city: str = "",
    region: str = "",
    latitude: float = 0,
    longitude: float = 0
):
    response = await diagnose(
        file=file,
        farmer_name=farmer_name,
        farm_name=farm_name,
        crop=crop,
        city=city,
        region=region,
        latitude=latitude,
        longitude=longitude
    )

    if response.status_code != 200:
        return response

    result = json.loads(response.body.decode("utf-8"))
    pdf_bytes = build_pdf_bytes(result)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=diagnosis_report.pdf"}
    )


# =========================
# FARMERS ENDPOINT
# =========================

@app.post("/farmers/register")
def register_farmer(
    name: str = Query(...),
    phone: str = Query(...),
    farm_name: str = Query(""),
    region: str = Query(...),
    city: str = Query(""),
    crop: str = Query("")
):
    try:
        save_farmer(name, phone, farm_name, region, city, crop)
        return {
            "success": True,
            "message": "تم تسجيل المزارع بنجاح"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# FORECAST ENDPOINT
# =========================

@app.get("/forecast/predict")
def forecast_predict(
    temperature: float = Query(...),
    humidity: float = Query(...),
    rainfall: float = Query(...),
    cases_count: float = Query(...),
    severity_avg: float = Query(...)
):
    try:
        base = forecast_ai_service.predict_cases(
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            cases_count=cases_count,
            severity_avg=severity_avg
        )

        risk_level_ar = base.get("risk_level_ar", "غير معروف")

        if risk_level_ar == "منخفض":
            risk_level_code = "low"
        elif risk_level_ar == "متوسط":
            risk_level_code = "moderate"
        elif risk_level_ar == "مرتفع":
            risk_level_code = "high"
        else:
            risk_level_code = "unknown"

        return {
            "predicted_cases": base.get("predicted_cases"),
            "risk_level_ar": risk_level_ar,
            "risk_level_code": risk_level_code,
            "features_used": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "cases_count": cases_count,
                "severity_avg": severity_avg
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# STATS ENDPOINTS
# =========================

@app.get("/stats/summary")
def stats_summary():
    conn = get_db_connection()

    total_cases = conn.execute("SELECT COUNT(*) as c FROM diagnoses").fetchone()["c"]
    avg_severity = conn.execute("SELECT AVG(severity_percent) as avg_s FROM diagnoses").fetchone()["avg_s"]

    if avg_severity is None:
        avg_severity = 0

    top_disease = conn.execute("""
        SELECT disease_ar, COUNT(*) as count
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY count DESC
        LIMIT 1
    """).fetchone()

    top_region = conn.execute("""
        SELECT region, COUNT(*) as count
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    return {
        "total_cases": total_cases,
        "avg_severity": round(float(avg_severity), 2),
        "top_disease": {
            "name": top_disease["disease_ar"] if top_disease else None,
            "count": top_disease["count"] if top_disease else 0
        },
        "top_region": {
            "name": top_region["region"] if top_region else None,
            "count": top_region["count"] if top_region else 0
        }
    }


@app.get("/stats/diseases")
def stats_diseases():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT disease_ar as disease, COUNT(*) as count
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/stats/regions")
def stats_regions():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT region, COUNT(*) as count
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/stats/severity")
def stats_severity():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT disease_ar as disease, AVG(severity_percent) as avg_severity
        FROM diagnoses
        GROUP BY disease_ar
        ORDER BY avg_severity DESC
    """).fetchall()
    conn.close()

    return [
        {
            "disease": r["disease"],
            "avg_severity": round(float(r["avg_severity"] or 0), 2)
        }
        for r in rows
    ]


# =========================
# MAP SUMMARY
# =========================

@app.get("/map/summary")
def map_summary():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT region, COUNT(*) as count, AVG(severity_percent) as avg_severity
        FROM diagnoses
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()
    conn.close()

    return [
        {
            "region": r["region"],
            "count": r["count"],
            "avg_severity": round(float(r["avg_severity"] or 0), 2)
        }
        for r in rows
    ]


# =========================
# ALERTS ENDPOINTS
# =========================

@app.get("/alerts/all")
def get_all_alerts():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT id, region, city, disease_name, crop, risk_score, risk_level,
               alert_message, recommendation, created_at
        FROM alerts
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/alerts/summary")
def alerts_summary():
    conn = get_db_connection()

    total_alerts = conn.execute("SELECT COUNT(*) as c FROM alerts").fetchone()["c"]
    high_alerts = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE risk_level='high'").fetchone()["c"]
    moderate_alerts = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE risk_level='moderate'").fetchone()["c"]

    latest = conn.execute("""
        SELECT region, disease_name, risk_score, risk_level, created_at
        FROM alerts
        ORDER BY created_at DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    return {
        "total_alerts": total_alerts,
        "high_alerts": high_alerts,
        "moderate_alerts": moderate_alerts,
        "latest_alert": dict(latest) if latest else None
    }


@app.get("/alerts/regions")
def alerts_regions():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT region, COUNT(*) as count
        FROM alerts
        GROUP BY region
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/alerts/high")
def alerts_high():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT id, region, city, disease_name, crop, risk_score, risk_level, created_at
        FROM alerts
        WHERE risk_level='high'
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/alerts/moderate")
def alerts_moderate():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT id, region, city, disease_name, crop, risk_score, risk_level, created_at
        FROM alerts
        WHERE risk_level='moderate'
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/export/alerts/json")
def export_alerts_json():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT *
        FROM alerts
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/alerts/send-region-sms")
def alerts_send_region_sms(
    region: str = Query(...),
    disease_name: str = Query(...),
    risk_score: float = Query(...)
):
    try:
        results = send_sms_to_region_farmers(region, disease_name, risk_score)
        return {
            "success": True,
            "region": region,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/alerts/create")
def create_alert(
    region: str = Query(...),
    city: str = Query(""),
    disease_name: str = Query(...),
    crop: str = Query(""),
    risk_score: float = Query(...),
    risk_level: str = Query(...),
    alert_message: str = Query(""),
    recommendation: str = Query("")
):
    try:
        save_alert(
            region=region,
            city=city,
            disease_name=disease_name,
            crop=crop,
            risk_score=risk_score,
            risk_level=risk_level,
            alert_message=alert_message,
            recommendation=recommendation
        )
        return {
            "success": True,
            "message": "تم حفظ التنبيه بنجاح"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# AUTH PLACEHOLDER
# =========================

@app.get("/auth/status")
def auth_status():
    return {
        "auth_enabled": False,
        "message": "صفحات login/register حالياً واجهات Frontend فقط ولم يتم ربط نظام مصادقة فعلي بعد."
    }


# =========================
# SYSTEM HEALTH
# =========================

@app.get("/system/health")
def system_health():
    return {
        "status": "running",
        "ai_model": "loaded",
        "forecast_model": "loaded" if forecast_ai_service.model is not None else "not_loaded",
        "database": "connected",
        "sms_configured": sms_service.is_configured(),
        "arabic_font_registered": AR_FONT_REGISTERED,
        "arabic_font_path": AR_FONT_PATH_USED,
        "version": "Phytologic AI v4.1"
    }
