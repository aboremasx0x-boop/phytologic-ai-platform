import sqlite3
import json
import os

DB_PATH = "data/diagnoses.db"

# =========================
# INIT DATABASE
# =========================

def init_db():
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS diagnoses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        farmer_name TEXT,
        farm_name TEXT,
        crop TEXT,

        plant TEXT,
        disease_class TEXT,
        disease_ar TEXT,
        confidence REAL,
        severity_percent REAL,
        cause TEXT,

        city TEXT,
        region TEXT,
        latitude REAL,
        longitude REAL,

        image_path TEXT,

        notes TEXT,
        question_answers TEXT,

        weather_snapshot TEXT,
        forecast_snapshot TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


# =========================
# GET CONNECTION
# =========================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# =========================
# SAVE DIAGNOSIS
# =========================

def save_diagnosis(
    farmer_name="",
    farm_name="",
    crop="",
    plant="",
    disease_class="",
    disease_ar="",
    confidence=0,
    severity_percent=0,
    cause="",
    city="",
    region="",
    latitude=None,
    longitude=None,
    image_path="",
    notes=None,
    question_answers=None,
    weather_snapshot=None,
    forecast_snapshot=None
):
    conn = get_connection()

    conn.execute("""
        INSERT INTO diagnoses (
            farmer_name, farm_name, crop,
            plant, disease_class, disease_ar,
            confidence, severity_percent, cause,
            city, region, latitude, longitude,
            image_path,
            notes, question_answers,
            weather_snapshot, forecast_snapshot
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farmer_name,
        farm_name,
        crop,

        plant,
        disease_class,
        disease_ar,

        confidence,
        severity_percent,
        cause,

        city,
        region,
        latitude,
        longitude,

        image_path,

        notes,
        json.dumps(question_answers) if question_answers else None,

        json.dumps(weather_snapshot) if weather_snapshot else None,
        json.dumps(forecast_snapshot) if forecast_snapshot else None
    ))

    conn.commit()
    conn.close()


# =========================
# GET ALL DIAGNOSES
# =========================

def get_all_diagnoses():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM diagnoses ORDER BY id DESC").fetchall()
    conn.close()

    results = []
    for r in rows:
        row = dict(r)

        # فك JSON
        try:
            row["question_answers"] = json.loads(row["question_answers"]) if row["question_answers"] else {}
        except:
            row["question_answers"] = {}

        try:
            row["weather_snapshot"] = json.loads(row["weather_snapshot"]) if row["weather_snapshot"] else {}
        except:
            row["weather_snapshot"] = {}

        try:
            row["forecast_snapshot"] = json.loads(row["forecast_snapshot"]) if row["forecast_snapshot"] else {}
        except:
            row["forecast_snapshot"] = {}

        results.append(row)

    return results


# =========================
# GET BY ID
# =========================

def get_diagnosis_by_id(diagnosis_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM diagnoses WHERE id = ?",
        (diagnosis_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    data = dict(row)

    try:
        data["question_answers"] = json.loads(data["question_answers"]) if data["question_answers"] else {}
    except:
        data["question_answers"] = {}

    try:
        data["weather_snapshot"] = json.loads(data["weather_snapshot"]) if data["weather_snapshot"] else {}
    except:
        data["weather_snapshot"] = {}

    try:
        data["forecast_snapshot"] = json.loads(data["forecast_snapshot"]) if data["forecast_snapshot"] else {}
    except:
        data["forecast_snapshot"] = {}

    return data


# =========================
# DELETE
# =========================

def delete_diagnosis(diagnosis_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM diagnoses WHERE id = ?", (diagnosis_id,))
    conn.commit()
    conn.close()


# =========================
# STATS (للداشبورد)
# =========================

def get_stats():
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as c FROM diagnoses").fetchone()["c"]

    avg_conf = conn.execute("SELECT AVG(confidence) as c FROM diagnoses").fetchone()["c"] or 0
    avg_sev = conn.execute("SELECT AVG(severity_percent) as s FROM diagnoses").fetchone()["s"] or 0

    conn.close()

    return {
        "total_cases": total,
        "avg_confidence": round(avg_conf, 2),
        "avg_severity": round(avg_sev, 2)
    }
