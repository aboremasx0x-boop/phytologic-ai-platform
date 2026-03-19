from typing import Dict, Any, List


# اربط الأمراض بالمحصول
# عدّل الأسماء حسب classes.json عندك
CROP_DISEASE_MAP = {
    "apple": [
        "Apple_Scab",
        "Apple_healthy",
        "Apple_rust",
        "Apple_black_rot",
        "Apple_Black_rot",
    ],
    "blueberry": [
        "Blueberry_healthy",
    ],
    "cherry": [
        "Cherry_healthy",
    ],
    "corn": [
        "Corn_Gray_leaf_spot",
        "Corn_rust_leaf",
        "Corn_leaf_blight",
    ],
    "grape": [
        "Grape_Black_rot",
        "Grape_healthy",
        "Grape_leaf_black_rot",
    ],
    "peach": [
        "Peach_healthy",
        "Peach_leaf",
    ],
    "pepper": [
        "Pepper_bell_Bacterial_spot",
        "Pepper_bell_healthy",
        "Bell_pepper_leaf",
        "Bell_pepper_leaf_spot",
    ],
    "potato": [
        "Potato_Early_blight",
        "Potato_Late_blight",
        "Potato_healthy",
        "Potato_leaf_early_blight",
        "Potato_leaf_late_blight",
    ],
    "raspberry": [
        "Raspberry_healthy",
    ],
    "soybean": [
        "Soybean_healthy",
        "Soyabean_healthy",
    ],
    "squash": [
        "Squash_Powdery_mildew",
    ],
    "strawberry": [
        "Strawberry_healthy",
        "Strawberry_leaf",
    ],
    "tomato": [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_mosaic_virus",
        "Tomato_Mosaic_virus",
        "Tomato_Tomato_YellowLeaf_Curl_Virus",
        "Tomato_YellowLeaf_Curl_Virus",
        "Tomato_healthy",
    ],
}


# أسئلة تشخيصية أولية
QUESTION_BANK = {
    "Tomato_Early_blight": [
        "هل توجد حلقات دائرية متداخلة داخل البقع؟",
        "هل الإصابة بدأت في الأوراق السفلية؟",
        "هل توجد بقع بنية تكبر تدريجيًا؟",
    ],
    "Tomato_Septoria_leaf_spot": [
        "هل البقع صغيرة ودائرية وحدودها داكنة؟",
        "هل مركز البقعة رمادي أو فاتح نسبيًا؟",
        "هل الإصابة شديدة في الأوراق السفلية؟",
    ],
    "Tomato_Bacterial_spot": [
        "هل البقع صغيرة ومائية في البداية؟",
        "هل توجد هالة صفراء حول بعض البقع؟",
        "هل السطح خشن أو متشقق في بعض المواضع؟",
    ],
    "Potato_Early_blight": [
        "هل توجد حلقات متداخلة داخل البقع؟",
        "هل البقع بنية وجافة نسبيًا؟",
    ],
    "Potato_Late_blight": [
        "هل البقع مائية وتمتد بسرعة؟",
        "هل توجد مناطق داكنة كبيرة على الورقة؟",
    ],
}


def normalize_crop_name(crop: str) -> str:
    crop = (crop or "").strip().lower()

    aliases = {
        "طماطم": "tomato",
        "tomato": "tomato",
        "بندورة": "tomato",
        "بطاطس": "potato",
        "بطاطا": "potato",
        "potato": "potato",
        "عنب": "grape",
        "grape": "grape",
        "تفاح": "apple",
        "apple": "apple",
        "فلفل": "pepper",
        "pepper": "pepper",
        "فراولة": "strawberry",
        "strawberry": "strawberry",
        "ذرة": "corn",
        "corn": "corn",
        "خوخ": "peach",
        "peach": "peach",
        "توت أزرق": "blueberry",
        "blueberry": "blueberry",
        "كرز": "cherry",
        "cherry": "cherry",
        "صويا": "soybean",
        "soybean": "soybean",
        "soyabean": "soybean",
        "اسكواش": "squash",
        "squash": "squash",
        "رازبيري": "raspberry",
        "raspberry": "raspberry",
    }
    return aliases.get(crop, crop)


def crop_matches_prediction(user_crop: str, predicted_class: str) -> bool:
    crop_key = normalize_crop_name(user_crop)
    if not crop_key:
        return True
    allowed = CROP_DISEASE_MAP.get(crop_key, [])
    return predicted_class in allowed


def get_questions_for_class(predicted_class: str) -> List[str]:
    return QUESTION_BANK.get(predicted_class, [
        "هل الأعراض موجودة بوضوح على الورقة المصابة؟",
        "هل الإصابة متكررة في أكثر من ورقة؟",
        "هل بدأت الإصابة في الأوراق السفلية أو القديمة؟",
    ])[:3]


def build_decision(
    best_confidence: float,
    second_confidence: float,
    quality_score: float,
    crop_match: bool,
    num_images: int
) -> Dict[str, Any]:
    diff = best_confidence - second_confidence

    score = (
        0.50 * best_confidence +
        0.20 * quality_score +
        0.20 * max(diff, 0) +
        0.10 * (100.0 if crop_match else 0.0)
    )

    rejection_reasons = []

    if num_images < 2:
        rejection_reasons.append("يفضل رفع صورتين إلى ثلاث صور لرفع موثوقية القرار")

    if quality_score < 45:
        rejection_reasons.append("جودة الصورة منخفضة")

    if best_confidence < 70:
        rejection_reasons.append("ثقة النموذج منخفضة")

    if diff < 8:
        rejection_reasons.append("هناك تشابه مرتفع بين التشخيص الأول والثاني")

    if not crop_match:
        rejection_reasons.append("المحصول المدخل لا يتوافق مع التشخيص المتوقع")

    if (
        quality_score < 45 or
        best_confidence < 70 or
        diff < 8 or
        not crop_match
    ):
        decision_status = "غير مؤكد"
        color = "red"
        action = "أعد التصوير"
    elif score >= 88:
        decision_status = "مؤكد"
        color = "green"
        action = "اعتمد النتيجة"
    else:
        decision_status = "متوسط"
        color = "yellow"
        action = "يفضل صورة إضافية أو مراجعة الأعراض"

    return {
        "final_score": round(score, 2),
        "decision_status": decision_status,
        "decision_color": color,
        "recommended_action": action,
        "rejection_reasons": rejection_reasons or ["لا يوجد"]
    }
