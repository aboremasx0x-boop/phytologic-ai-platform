import os
import sqlite3
from datetime import datetime

BASE_DIR = os.getenv("DATA_DIR", ".")
DB_PATH = os.path.join(BASE_DIR, "diagnoses.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_diagnoses_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
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
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def create_alerts_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region TEXT,
        city TEXT,
        disease_name TEXT,
        crop TEXT,
        risk_score REAL,
        risk_level TEXT,
        alert_message TEXT,
        recommendation TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def create_farmers_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        farm_name TEXT,
        region TEXT,
        city TEXT,
        crop TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def init_db():
    os.makedirs(BASE_DIR, exist_ok=True)
    create_diagnoses_table()
    create_alerts_table()
    create_farmers_table()


def save_diagnosis(
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
    image_path
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO diagnoses (
        farmer_name, farm_name, crop, plant, disease_class, disease_ar,
        confidence, severity_percent, cause, city, region,
        latitude, longitude, image_path, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def save_alert(
    region,
    city,
    disease_name,
    crop,
    risk_score,
    risk_level,
    alert_message,
    recommendation
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO alerts (
        region, city, disease_name, crop, risk_score,
        risk_level, alert_message, recommendation, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        region,
        city,
        disease_name,
        crop,
        risk_score,
        risk_level,
        alert_message,
        recommendation,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def save_farmer(name, phone, farm_name, region, city, crop):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO farmers (
        name, phone, farm_name, region, city, crop, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        phone,
        farm_name,
        region,
        city,
        crop,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()
