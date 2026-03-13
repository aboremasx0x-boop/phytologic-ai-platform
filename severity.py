# severity.py
# (نسخة محسّنة بدون حذف أي وظيفة موجودة — تحسين الحساب فقط + إضافة أقسام للتوصيات)

from typing import Dict, Any, List
import numpy as np
from PIL import Image


# ---------------------------------------
# أدوات مساعدة: قصّ الورقة (تقريبي) لتفادي حساب الخلفية
# ---------------------------------------
def _to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def _leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """
    قناع تقريبي للورقة اعتماداً على أن الورقة غالباً تميل للأخضر.
    الهدف: حساب شدة الإصابة داخل الورقة فقط بدل الخلفية.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # ورقة خضراء غالباً: g أعلى من r و b بفارق بسيط
    m1 = (g > 40) & (g >= r + 10) & (g >= b + 10)

    # استبعاد شديد البياض (خلفية مضيئة)
    m2 = ~((r > 230) & (g > 230) & (b > 230))

    leaf = m1 & m2

    # لو القناع صغير جدًا (صورة غير ورقة/إضاءة سيئة) نعمل fallback: اعتبر كل الصورة "ورقة"
    if leaf.mean() < 0.05:
        leaf = np.ones((rgb.shape[0], rgb.shape[1]), dtype=bool)

    return leaf


def _lesion_mask(rgb: np.ndarray) -> np.ndarray:
    """
    قناع تقريبي للبقع المرضية (داكن/بني/مصفر).
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # (1) بقع داكنة/بنية: g منخفض نسبيًا + r متوسط
    m_brown = (g < 140) & (r > 60) & (b < 150)

    # (2) اصفرار/كلوروز: r و g مرتفعين و b أقل (مصفر)
    m_yellow = (r > 140) & (g > 140) & (b < 140)

    return m_brown | m_yellow


# ---------------------------------------
# حساب نسبة الإصابة تقريبياً من الصورة
# ---------------------------------------
def _estimate_severity_percent(pil_img: Image.Image) -> float:
    """
    تقدير نسبة الإصابة اعتماداً على لون البقع
    (طريقة تقريبية وليست قياساً مخبرياً).

    تحسين مهم: الحساب يتم داخل "قناع الورقة" لتقليل تأثير الخلفية.
    """
    rgb = _to_np_rgb(pil_img)

    leaf = _leaf_mask(rgb)
    lesion = _lesion_mask(rgb)

    # احسب نسبة الآفة داخل الورقة فقط
    leaf_area = float(leaf.sum())
    if leaf_area <= 0:
        return 0.0

    lesion_in_leaf = float((lesion & leaf).sum())
    percent = (lesion_in_leaf / leaf_area) * 100.0

    # قصّ للنطاق
    if percent < 0:
        percent = 0.0
    if percent > 100:
        percent = 100.0

    return round(float(percent), 2)


# ---------------------------------------
# تحويل النسبة إلى مستوى إصابة
# ---------------------------------------
def _severity_level_from_percent(p: float) -> str:
    if p < 7:
        return "low"
    elif p < 20:
        return "medium"
    else:
        return "high"


# ---------------------------------------
# تسمية المستوى عربي / إنجليزي
# ---------------------------------------
def _severity_label(level: str, lang: str) -> Dict[str, str]:
    ar = {"low": "منخفضة", "medium": "متوسطة", "high": "شديدة"}
    en = {"low": "Low", "medium": "Medium", "high": "High"}

    return {
        "ar": ar.get(level, level),
        "en": en.get(level, level),
    }


# ---------------------------------------
# توصيات المكافحة (IPM) — (قديمة) قائمة واحدة
# ---------------------------------------
def _recommendations(pred_class: str, level: str, lang: str) -> List[str]:
    # نحافظ على نفس الإخراج السابق (List[str]) حتى لا تتعطل الواجهة
    if (lang or "ar").lower().startswith("ar"):
        rec = [
            "إزالة الأوراق المصابة من النبات.",
            "تحسين التهوية بين النباتات.",
            "تجنب بلل الأوراق أثناء الري.",
            "التخلص من بقايا النباتات المصابة.",
        ]
        if level == "high":
            rec.insert(0, "الإصابة مرتفعة: يفضل عزل النبات المصاب لمنع انتشار المرض.")
        return rec
    else:
        rec = [
            "Remove infected leaves.",
            "Improve plant spacing and airflow.",
            "Avoid wetting foliage during irrigation.",
            "Remove infected plant debris.",
        ]
        if level == "high":
            rec.insert(0, "High severity: isolate infected plants.")
        return rec


# ---------------------------------------
# توصيات جديدة مُقسّمة 3 أقسام (إضافة فقط)
# ---------------------------------------
def _recommendations_sections(pred_class: str, level: str, lang: str) -> Dict[str, List[str]]:
    ar = (lang or "ar").lower().startswith("ar")

    if ar:
        bio = [
            "مكافحة حيوية: استخدام عوامل حيوية مسجلة/متاحة محلياً حسب النظام (مثل منتجات Trichoderma أو Bacillus) وفق بطاقة المنتج.",
            "مكافحة حيوية: دعم الأحياء النافعة بتقليل الرطوبة على الأوراق وتحسين التهوية.",
        ]
        agr = [
            "إدارة زراعية: إزالة الأوراق/الأجزاء المصابة والتخلص منها خارج المزرعة.",
            "إدارة زراعية: تجنب الري بالرش على المجموع الخضري، والري صباحاً إن لزم.",
            "إدارة زراعية: ترك مسافات/تقليم لتحسين حركة الهواء وتقليل البلل.",
        ]
        pest = [
            "مبيدات (عام): في حال الحاجة، استخدم مبيداً مسجلاً للمحصول والمرض في بلدك واتبع بطاقة المنتج وفترات الأمان.",
            "مبيدات (عام): بدّل بين مجموعات فعّالة مختلفة لتقليل المقاومة (Rotation).",
        ]

        if level == "high":
            agr.insert(0, "إدارة زراعية (عاجل): عزل/تمييز النباتات الأعلى إصابة لمنع الانتشار.")

    else:
        bio = [
            "Biocontrol: use locally registered biocontrol products (e.g., Trichoderma / Bacillus-based) according to the label.",
            "Biocontrol: support beneficial microbes by reducing leaf wetness and improving airflow.",
        ]
        agr = [
            "Agronomic: remove infected leaves/plant parts and dispose outside the field.",
            "Agronomic: avoid overhead irrigation; irrigate early if needed.",
            "Agronomic: spacing/pruning to improve airflow and reduce leaf wetness.",
        ]
        pest = [
            "Pesticides (general): if needed, use a locally registered product for crop/disease and follow label & PHI.",
            "Pesticides (general): rotate modes of action to reduce resistance.",
        ]

        if level == "high":
            agr.insert(0, "Agronomic (urgent): isolate/mark highly infected plants to reduce spread.")

    return {
        "biocontrol": bio,
        "agronomic": agr,
        "pesticides_general": pest,
    }


def _flatten_sections(sections: Dict[str, List[str]]) -> List[str]:
    # ترتيب ثابت للعرض
    out: List[str] = []
    out.extend(sections.get("biocontrol", []))
    out.extend(sections.get("agronomic", []))
    out.extend(sections.get("pesticides_general", []))
    return out


# ---------------------------------------
# الدالة الأساسية المستخدمة في predict.py
# ---------------------------------------
def estimate_severity_and_recommendations(
    pil_img: Image.Image,
    pred_class: str,
    lang: str = "ar"
) -> Dict[str, Any]:
    p = _estimate_severity_percent(pil_img)
    level = _severity_level_from_percent(p)

    sections = _recommendations_sections(pred_class, level, lang)

    # IMPORTANT:
    # - نُبقي "recommendations" كما كانت (List) حتى لا تختفي في index
    # - ونضيف "recommendations_sections" كميزة جديدة (اختياري عرضها لاحقًا)
    return {
        "severity_level": level,
        "severity_percent_est": p,
        "label": _severity_label(level, lang),

        # القديم (لا نحذفه)
        "recommendations": _recommendations(pred_class, level, lang),

        # الجديد (إضافة)
        "recommendations_sections": sections,
        "recommendations_all": _flatten_sections(sections),
    }


# ---------------------------------------
# اسم بديل احتياطي
# ---------------------------------------
def estimate_severity(
    pil_img: Image.Image,
    pred_class: str,
    lang: str = "ar"
) -> Dict[str, Any]:
    return estimate_severity_and_recommendations(pil_img, pred_class, lang)
