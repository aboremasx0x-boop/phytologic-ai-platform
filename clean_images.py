import os
from PIL import Image

# ضع هنا مجلد البيانات الذي يحتوي على الصور
DATA_DIR = "data_v3"

deleted = 0
checked = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        path = os.path.join(root, file)

        # نتأكد أنه ملف صورة
        if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")):
            checked += 1
            try:
                img = Image.open(path)
                img.verify()   # يتحقق من سلامة الصورة
            except Exception:
                print("Deleted corrupted image:", path)
                os.remove(path)
                deleted += 1

print("\nDone")
print("Checked images:", checked)
print("Deleted images:", deleted)
