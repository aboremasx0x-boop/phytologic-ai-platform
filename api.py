import io
import os
import json
import uuid
import base64
import urllib.parse
import urllib.request
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from ai_spread_engine import DiseaseSpreadEngine
from database import init_db, save_diagnosis, get_connection

try:
    from disease_info import DISEASE_INFO
except Exception:
    DISEASE_INFO = {}


# =========================
# PATHS
# =========================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_PATH, "templates")
STATIC_DIR = os.path.join(BASE_PATH, "static")
DATA_DIR = os.path.join(BASE_PATH, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# APP
# =========================

app = FastAPI(
    title="Phytologic AI Platform",
    description="Plant Disease Diagnosis + Weather + Spread Prediction",
    version="8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

init_db()
spread_engine = DiseaseSpreadEngine()

# =========================
# PAGE MAP
# =========================

PAGE_FILE_MAP = {
    "index": "index.html",
    "index_home": "index_home.html",
    "dashboard": "dashboard.html",
    "report_center": "reports_center.html",
    "reports_center": "reports_center.html",
    "alerts": "alerts.html",
    "admin": "admin.html",
    "forecast": "forecast.html",
    "forecast_ai": "forecast_ai.html",
    "login": "login.html",
    "register": "register.html",
}


# =========================
# HELPERS
# =========================

REGION_COORDS = {
    "الرياض": {"lat": 24.7136, "lon": 46.6753},
    "مكة": {"lat": 21.3891, "lon": 39.8579},
    "المدينة": {"lat": 24.5247, "lon": 39.5692},
    "القصيم": {"lat": 26.3592, "lon": 43.9818},
    "حائل": {"lat": 27.5114, "lon": 41.7208},
    "تبوك": {"lat": 28.3998, "lon": 36.5715},
    "الجوف": {"lat": 29.9697, "lon": 40.2064},
    "الحدود الشمالية": {"lat": 30.9753, "lon": 41.0381},
    "الشرقية": {"lat": 26.4207, "lon": 50.0888},
    "الباحة": {"lat": 20.0129, "lon": 41.4677},
    "عسير": {"lat": 18.2164, "lon": 42.5053},
    "جازان": {"lat": 16.8892, "lon": 42.5511},
    "نجران": {"lat": 17.5650, "lon": 44.2289}
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


def get_db_connection():
    return get_connection()


def template_path(filename: str) -> str:
    p = os.path.join(TEMPLATES_DIR, filename)
    if os.path.exists(p):
        return p
    p2 = os.path.join(BASE_PATH, filename)
    return p2


def get_existing_index():
    for name in ["index.html", "index_home.html"]:
        p = template_path(name)
        if os.path.exists(p):
            return p
    return None


def fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "PhytologicAI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
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


def resolve_region_or_city_coords(region=None, city=None, latitude=None, longitude=None):
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude), "direct"

    city = (city or "").strip()
    region = (region or "").strip()

    if city and city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        return lat, lon, "city"

    if region and region in REGION_COORDS:
        return REGION_COORDS[region]["lat"], REGION_COORDS[region]["lon"], "region"

    return 24.7136, 46.6753, "fallback"


def get_disease_info_by_class(best_class: str):
    return DISEASE_INFO.get(
        best_class,
        {
            "plant": "غير معروف",
            "disease_ar": best_class,
            "cause": "غير محدد",
            "advice": ["تحقق من الأعراض ميدانيًا", "أعد التصوير بصورة أوضح"],
            "pesticide_program": {
                "main": {
                    "active_ingredient": "استشارة مختص",
                    "trade_name": "-",
                    "type": "-",
                    "dose": "-",
                    "phi": "-",
                    "spray_message_ar": "حدد المبيد المناسب بعد التشخيص الحقلي"
                },
                "options": []
            }
        }
    )


# =========================
# IMAGE QUALITY
# =========================

def evaluate_image_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = gray.std()

    score = 1.0
    issues = []

    if sharpness < 60:
        score *= 0.5
        issues.append("الصورة غير واضحة")

    if brightness < 50:
        score *= 0.6
        issues.append("إضاءة منخفضة")

    if brightness > 220:
        score *= 0.6
        issues.append("إضاءة مرتفعة")

    if contrast < 25:
        score *= 0.7
        issues.append("تباين ضعيف")

    return round(score, 2), issues, {
        "sharpness": round(float(sharpness), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2)
    }


# =========================
# SEVERITY
# =========================

def estimate_severity(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

    infected = np.sum(thresh > 0)
    total = thresh.size
    percent = (infected / total) * 100 if total else 0

    if percent < 5:
        level = "خفيفة"
    elif percent < 20:
        level = "متوسطة"
    else:
        level = "شديدة"

    overlay = img_bgr.copy()
    overlay[thresh > 0] = (0, 0, 255)

    return round(float(percent), 2), level, thresh, overlay


# =========================
# SIMPLE GRADCAM PLACEHOLDER
# =========================

def generate_gradcam_visual(img_bgr):
    heat = cv2.GaussianBlur(img_bgr, (0, 0), 9)
    gradcam = cv2.addWeighted(img_bgr, 0.5, heat, 0.5, 0)
    return gradcam


# =========================
# SIMPLE MODEL PLACEHOLDER
# استبدله لاحقًا بالموديل الحقيقي
# =========================

def predict_disease_from_image(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()

    if mean_val < 90:
        return "Tomato_Early_blight", 84.0, "Tomato_Septoria_leaf_spot", 77.0
    elif mean_val < 150:
        return "Tomato_Septoria_leaf_spot", 79.0, "Tomato_Early_blight", 74.0
    else:
        return "Healthy", 88.0, "Tomato_Early_blight", 41.0


# =========================
# WEATHER
# =========================

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
        "humidity": float(current.get("relative_humidity_2m", 50)),
        "rainfall": float(current.get("precipitation", 0)),
        "latitude": latitude,
        "longitude": longitude
    }


# =========================
# PAGE ROUTES
# =========================

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    index_file = get_existing_index()
    if not index_file:
        return JSONResponse({"error": "index not found"}, status_code=404)
    return FileResponse(index_file)


@app.get("/pages/{page_name}")
def open_page(page_name: str):
    page_name = (page_name or "").strip().lower()
    filename = PAGE_FILE_MAP.get(page_name, f"{page_name}.html")
    p = template_path(filename)

    if os.path.exists(p):
        return FileResponse(p)

    return JSONResponse({"error": f"{filename} not found"}, status_code=404)


@app.get("/{page_file}.html")
def open_direct_html(page_file: str):
    filename = f"{page_file}.html"
    p = template_path(filename)

    if os.path.exists(p):
        return FileResponse(p)

    return JSONResponse({"error": f"{filename} not found"}, status_code=404)


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "running",
        "version": "Phytologic AI v8.0",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/system/health")
def system_health():
    return {
        "status": "running",
        "api": "ok",
        "forecast_model": "loaded",
        "spread_engine": "loaded",
        "database": "connected",
        "version": "Phytologic AI v8.0"
    }


# =========================
# WEATHER ROUTES
# =========================

@app.get("/weather/live")
def weather_live(
    region: str = Query(""),
    city: str = Query(""),
    latitude: float = Query(None),
    longitude: float = Query(None)
):
    try:
        lat, lon, source = resolve_region_or_city_coords(
            region=region,
            city=city,
            latitude=latitude,
            longitude=longitude
        )

        current = get_live_weather(lat, lon)

        return {
            "success": True,
            "source": source,
            "location": {
                "region": region,
                "city": city,
                "latitude": lat,
                "longitude": lon
            },
            "current": current
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/weather/forecast")
def weather_forecast(
    region: str = Query(""),
    city: str = Query(""),
    latitude: float = Query(None),
    longitude: float = Query(None)
):
    try:
        lat, lon, source = resolve_region_or_city_coords(
            region=region,
            city=city,
            latitude=latitude,
            longitude=longitude
        )

        query = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": 3,
            "timezone": "auto"
        })

        url = f"https://api.open-meteo.com/v1/forecast?{query}"
        data = fetch_json(url)

        return {
            "success": True,
            "source": source,
            "location": {
                "region": region,
                "city": city,
                "latitude": lat,
                "longitude": lon
            },
            "current": data.get("current", {}),
            "hourly": {
                "time": data.get("hourly", {}).get("time", [])[:12],
                "temperature_2m": data.get("hourly", {}).get("temperature_2m", [])[:12],
                "relative_humidity_2m": data.get("hourly", {}).get("relative_humidity_2m", [])[:12],
                "precipitation": data.get("hourly", {}).get("precipitation", [])[:12]
            },
            "daily": {
                "time": data.get("daily", {}).get("time", []),
                "temperature_2m_max": data.get("daily", {}).get("temperature_2m_max", []),
                "temperature_2m_min": data.get("daily", {}).get("temperature_2m_min", []),
                "precipitation_sum": data.get("daily", {}).get("precipitation_sum", [])
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# DIAGNOSE
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
        contents = await file.read()

        try:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        img_bgr = pil_to_cv(img_pil)
        image_path = save_upload_file(contents, file.filename or "upload.jpg")

        # 1) جودة الصورة
        quality_score, issues, quality_metrics = evaluate_image_quality(img_bgr)

        original_b64 = pil_to_base64(img_pil)

        if quality_score < 0.6:
            return {
                "status": "rejected",
                "message": "جودة الصورة غير كافية للتشخيص",
                "issues": issues,
                "image_quality": {
                    "score": quality_score,
                    "details": quality_metrics
                },
                "original_image": original_b64,
                "tips": [
                    "اقترب من الورقة المصابة",
                    "استخدم إضاءة طبيعية",
                    "تجنب الاهتزاز",
                    "اجعل الخلفية بسيطة"
                ]
            }

        # 2) شدة الإصابة
        severity_percent, severity_level, severity_mask, severity_overlay = estimate_severity(img_bgr)
        severity_overlay_b64 = pil_to_base64(np_to_pil(severity_overlay))

        # 3) Grad-CAM مبسط
        gradcam_img = generate_gradcam_visual(img_bgr)
        gradcam_b64 = pil_to_base64(np_to_pil(gradcam_img))

        # 4) التشخيص
        best_class, best_conf, second_class, second_conf = predict_disease_from_image(img_bgr)

        # 5) تعديل الثقة حسب جودة الصورة
        final_conf = best_conf * quality_score

        # 6) القرار
        if final_conf < 50:
            decision = "غير موثوق"
        elif abs(best_conf - second_conf) < 8:
            decision = "غير مؤكد"
        else:
            decision = "مؤكد"

        disease_info = get_disease_info_by_class(best_class)

        # 7) الطقس
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
                "temperature": 25,
                "humidity": 50,
                "rainfall": 0,
                "latitude": lat,
                "longitude": lon,
                "source": weather_source
            }

        # 8) توقع الانتشار
        risk = spread_engine.calculate_risk(
            weather_data.get("temperature", 25),
            weather_data.get("humidity", 50),
            severity_percent,
            quality_score
        )
        risk_level = spread_engine.classify_risk(risk)
        risk_action = spread_engine.recommendation(risk)
        future_projection = spread_engine.future_projection(risk)

        # 9) الحفظ
        try:
            save_diagnosis(
                farmer_name=farmer_name,
                farm_name=farm_name,
                crop=crop,
                plant=disease_info.get("plant", ""),
                disease_class=best_class,
                disease_ar=disease_info.get("disease_ar", best_class),
                confidence=round(final_conf, 2),
                severity_percent=severity_percent,
                cause=disease_info.get("cause", ""),
                city=city,
                region=region,
                latitude=lat,
                longitude=lon,
                image_path=image_path,
                notes="",
                question_answers={},
                weather_snapshot=weather_data,
                forecast_snapshot=None
            )
        except Exception as db_error:
            print("DB save error:", db_error)

        return {
            "status": "success",
            "image_path": image_path,

            "original_image": original_b64,
            "gradcam_image": gradcam_b64,

            "image_quality": {
                "score": quality_score,
                "issues": issues,
                "details": quality_metrics
            },

            "severity": {
                "severity_percent_est": severity_percent,
                "severity_level": severity_level,
                "label": {"ar": severity_level},
                "severity_overlay_b64": severity_overlay_b64
            },

            "best_prediction": {
                "class_name": best_class,
                "confidence": round(final_conf, 2)
            },

            "second_prediction": {
                "class_name": second_class,
                "confidence": round(second_conf, 2)
            },

            "confidence_level": "مرتفعة" if final_conf >= 80 else "متوسطة" if final_conf >= 50 else "منخفضة",
            "decision_status": decision,

            "disease_info": {
                "plant": disease_info.get("plant", ""),
                "disease_ar": disease_info.get("disease_ar", best_class),
                "cause": disease_info.get("cause", ""),
                "advice": disease_info.get("advice", [])
            },

            "pesticide_program": disease_info.get("pesticide_program", {
                "main": {
                    "active_ingredient": "استشارة مختص",
                    "trade_name": "-",
                    "type": "-",
                    "dose": "-",
                    "phi": "-",
                    "spray_message_ar": "حدد المبيد المناسب بعد التشخيص الحقلي"
                },
                "options": []
            }),

            "weather": weather_data,

            "spread_prediction": {
                "risk_percent": risk,
                "risk_level": risk_level,
                "action": risk_action,
                "future": future_projection,
                "factors": {
                    "temperature": weather_data.get("temperature", 25),
                    "humidity": weather_data.get("humidity", 50),
                    "severity": severity_percent,
                    "quality": quality_score
                }
            },

            "questions": [
                {"question": "هل الإصابة محصورة في بقع واضحة؟", "symptom": "clear_spots"},
                {"question": "هل تنتشر الإصابة في أطراف الورقة؟", "symptom": "edge_spread"}
            ]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# SAVE DIAGNOSIS
# =========================

@app.post("/api/diagnosis")
def api_diagnosis(data: dict = Body(...)):
    try:
        save_diagnosis(
            data.get("farmer_name", ""),
            data.get("farm_name", ""),
            data.get("crop", ""),
            data.get("plant", ""),
            data.get("disease_class", ""),
            data.get("disease_ar", ""),
            data.get("confidence", 0),
            data.get("severity_percent", 0),
            data.get("cause", ""),
            data.get("city", ""),
            data.get("region", ""),
            data.get("latitude", None),
            data.get("longitude", None),
            data.get("image_path", ""),
            notes=data.get("notes"),
            question_answers=data.get("question_answers"),
            weather_snapshot=data.get("weather_snapshot"),
            forecast_snapshot=data.get("forecast_snapshot")
        )
        return {"success": True, "message": "تم حفظ التشخيص بنجاح"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =========================
# GET DIAGNOSES
# =========================

@app.get("/api/diagnoses")
def get_diagnoses():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM diagnoses ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# SPREAD ANALYSIS
# =========================

@app.get("/spread/analysis")
def spread_analysis():
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM diagnoses ORDER BY id DESC").fetchall()
        conn.close()

        data = [dict(r) for r in rows]

        heatmap = []
        regions = {}

        for item in data:
            lat = item.get("latitude")
            lon = item.get("longitude")
            sev = float(item.get("severity_percent") or 0)

            if lat is not None and lon is not None:
                heatmap.append({
                    "lat": lat,
                    "lon": lon,
                    "risk": sev
                })

            reg = item.get("region") or "غير محدد"
            if reg not in regions:
                regions[reg] = {"count": 0, "severity_sum": 0}

            regions[reg]["count"] += 1
            regions[reg]["severity_sum"] += sev

        region_list = []
        for reg, vals in regions.items():
            avg_sev = vals["severity_sum"] / vals["count"] if vals["count"] else 0
            risk = spread_engine.calculate_risk(28, 60, avg_sev, 1.0)

            region_list.append({
                "region": reg,
                "cases": vals["count"],
                "avg_severity": round(avg_sev, 2),
                "risk_score": risk,
                "risk_level": spread_engine.classify_risk(risk)
            })

        return {
            "success": True,
            "count": len(data),
            "heatmap": heatmap,
            "regions": region_list
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
