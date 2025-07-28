import os
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ====== Config ======
MODEL_PATH = "D:/Auto_aiops/checkpoints/best_effnet_b3_multilabel.pth"
IMAGE_DIR = "D:/inference_images"
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    'bridges_any', 'bridges_damage',
    'buildings_affected', 'buildings_any', 'buildings_destroyed',
    'buildings_major', 'buildings_minor',
    'debris_any',
    'flooding_any', 'flooding_structures',
    'roads_any', 'roads_damage',
    'trees_any', 'trees_damage',
    'water_any'
]
NUM_CLASSES = len(LABELS)

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ====== Load Model ======
print(f"\n🔁 Loading model from: {MODEL_PATH}")
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)

# Load checkpoint state_dict
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle classifier mismatch
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]  # If wrapped in a dict

# Strip weights if classifier doesn't match
classifier_weight = checkpoint.get("classifier.weight", None)
if classifier_weight is not None and classifier_weight.shape[0] != NUM_CLASSES:
    print(f"⚠️ Classifier mismatch: checkpoint has {classifier_weight.shape[0]} classes, expected {NUM_CLASSES}")
    del checkpoint["classifier.weight"]
    del checkpoint["classifier.bias"]

# Load partial weights
missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"✅ Model loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

model.to(DEVICE)
model.eval()

# ====== Inference ======
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = [LABELS[i] for i, p in enumerate(probs) if p > 0.5]
    
    return predictions, probs

# ====== Run Inference ======
print(f"\n📂 Running inference on images in: {IMAGE_DIR}\n")

for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(IMAGE_DIR, img_name)
        labels, probs = predict_image(path)

        print(f"🖼️ {img_name}")
        for i, label in enumerate(LABELS):
            print(f"   - {label:<25}: {probs[i]:.3f} {'✅' if label in labels else ''}")
        print()


    