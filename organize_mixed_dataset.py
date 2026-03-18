import os
import shutil
from pathlib import Path

DATASET_ROOT = Path("data_v3")
OUTPUT_ROOT = Path("data_v3_organized")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NAME_MAP = {
    "tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "tomato_bacterial_spot": "Tomato_Bacterial_spot",
    "tomato leaf late blight": "Tomato_Late_blight",
    "tomato_late_blight": "Tomato_Late_blight",
    "tomato leaf mosaic virus": "Tomato_Mosaic_virus",
    "tomato_tomato_mosaic_virus": "Tomato_Mosaic_virus",
    "tomato leaf yellow virus": "Tomato_YellowLeaf_Curl_Virus",
    "tomato_tomato_yellowleaf_curl_virus": "Tomato_YellowLeaf_Curl_Virus",
    "tomato mold leaf": "Tomato_Leaf_Mold",
    "tomato_leaf_mold": "Tomato_Leaf_Mold",
    "tomato septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "tomato_septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "tomato early blight leaf": "Tomato_Early_blight",
    "tomato_early_blight": "Tomato_Early_blight",
    "tomato target spot": "Tomato_Target_Spot",
    "tomato_target_spot": "Tomato_Target_Spot",
    "tomato healthy": "Tomato_healthy",
    "tomato_healthy": "Tomato_healthy",
    "tomato_spider_mites_two_spotted_spider_mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "potato leaf early blight": "Potato_Early_blight",
    "potato_early_blight": "Potato_Early_blight",
    "potato leaf late blight": "Potato_Late_blight",
    "potato_late_blight": "Potato_Late_blight",
    "potato_healthy": "Potato_healthy",
    "apple scab leaf": "Apple_Scab",
    "apple rust leaf": "Apple_rust",
    "apple leaf": "Apple_healthy",
    "bell_pepper leaf spot": "Pepper_bell_Bacterial_spot",
    "pepper_bell_bacterial_spot": "Pepper_bell_Bacterial_spot",
    "bell_pepper leaf": "Pepper_bell_healthy",
    "pepper_bell_healthy": "Pepper_bell_healthy",
    "corn gray leaf spot": "Corn_Gray_leaf_spot",
    "corn leaf blight": "Corn_leaf_blight",
    "corn rust leaf": "Corn_rust_leaf",
    "grape leaf black rot": "Grape_Black_rot",
    "grape leaf": "Grape_healthy",
    "peach leaf": "Peach_healthy",
    "blueberry leaf": "Blueberry_healthy",
    "cherry leaf": "Cherry_healthy",
    "raspberry leaf": "Raspberry_healthy",
    "soyabean leaf": "Soybean_healthy",
    "squash powdery mildew leaf": "Squash_Powdery_mildew",
    "strawberry leaf": "Strawberry_healthy",
}

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text

def detect_source_from_name(folder_name: str) -> str:
    name = folder_name.lower()
    if "_" in folder_name and folder_name.count("_") >= 1:
        return "plantvillage"
    return "plantdoc"

def canonical_name(folder_name: str) -> str:
    key = normalize_text(folder_name)
    if key in NAME_MAP:
        return NAME_MAP[key]
    return folder_name.replace(" ", "_")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def unique_dest(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def main():
    if not DATASET_ROOT.exists():
        print("المجلد data_v3 غير موجود")
        return

    ensure_dir(OUTPUT_ROOT)

    folders = [f for f in DATASET_ROOT.iterdir() if f.is_dir()]
    if not folders:
        print("لا توجد مجلدات داخل data_v3")
        return

    total_images = 0

    for folder in folders:
        original_folder_name = folder.name
        source = detect_source_from_name(original_folder_name)
        disease_name = canonical_name(original_folder_name)

        target_dir = OUTPUT_ROOT / disease_name / source
        ensure_dir(target_dir)

        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                dest = unique_dest(target_dir / file.name)
                shutil.copy2(file, dest)
                total_images += 1

    print("تم الانتهاء بنجاح")
    print("عدد الصور المنقولة:", total_images)
    print("المجلد الجديد:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
