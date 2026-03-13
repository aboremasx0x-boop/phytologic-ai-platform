# ===== إصلاح عرض العربية وحفظ النتائج بشكل صحيح =====
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import torch
from torchvision import transforms, models
from PIL import Image
from disease_info import DISEASE_INFO

# =========================
# إعدادات الملفات
# =========================

MODEL_PATH = "plant_disease_model_v3.pth"
IMAGE_PATH = "test.jpg"
RESULT_PATH = "result.txt"

# =========================
# تحميل النموذج
# =========================

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

classes = checkpoint["classes"]
num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# =========================
# تجهيز الصورة
# =========================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# =========================
# التحقق من وجود الصورة
# =========================

if not os.path.exists(IMAGE_PATH):
    print(f"خطأ: الصورة غير موجودة: {IMAGE_PATH}")
    print("ضع صورة داخل مجلد المشروع باسم test.jpg")
    raise SystemExit

img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0)

# =========================
# التنبؤ
# =========================

with torch.no_grad():
    out = model(img)
    probs = torch.softmax(out, dim=1)
    top_probs, top_idx = torch.topk(probs, k=3)

# =========================
# تجهيز النصوص
# =========================

lines = []
lines.append("أفضل 3 احتمالات:")

for p, i in zip(top_probs[0], top_idx[0]):
    cls = classes[i.item()]
    lines.append(f"{cls} : {p.item()*100:.2f}%")

best_class = classes[top_idx[0][0].item()]

if best_class in DISEASE_INFO:
    info = DISEASE_INFO[best_class]

    lines.append("\n--- معلومات التشخيص ---")
    lines.append(f"النبات: {info['plant']}")
    lines.append(f"المرض: {info['disease_ar']}")
    lines.append(f"المسبب: {info['cause']}")
    lines.append("التوصيات:")

    for item in info["advice"]:
        lines.append(f"- {item}")
else:
    lines.append("\nلا توجد معلومات مضافة لهذه الفئة بعد.")

# =========================
# طباعة النتائج
# =========================

for line in lines:
    print(line)

# =========================
# حفظ النتائج في ملف نصي
# =========================

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"\nتم حفظ النتيجة في ملف {RESULT_PATH}")
