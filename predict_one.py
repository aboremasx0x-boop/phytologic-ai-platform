import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image

# ====== Paths ======
MODEL_PATH = "plant_disease_model.pth"
DATA_DIR = "data"
IMAGE_PATH = "test.jpg"   # غيّرها إذا اسم/مكان الصورة مختلف

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====== Get classes exactly like training ======
tmp_dataset = datasets.ImageFolder(DATA_DIR)
classes = tmp_dataset.classes
num_classes = len(classes)
print("Classes:", classes)

# ====== Transform (same as training) ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====== Load model architecture ======
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ====== Load weights ======
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

# ====== Load + predict ======
img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0).to(device)  # shape: [1,3,224,224]

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

pred_idx = int(probs.argmax())
pred_class = classes[pred_idx]
pred_prob = float(probs[pred_idx])

print("\nPrediction:")
print(" - Class:", pred_class)
print(" - Confidence:", round(pred_prob * 100, 2), "%")

print("\nAll probabilities:")
for c, p in sorted(zip(classes, probs), key=lambda t: t[1], reverse=True):
    print(f" - {c}: {p*100:.2f}%")