import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ====== Config ======
MODEL_PATH = "D:/GitHub/auto_ai_ops/best_effnet_b3_multilabel.pth"
IMAGE_DIR = "D:/ladidataset/ladi_v2_resized/v2/images_resized"
METADATA_PATH = "D:/ladidataset/ladi_v2_resized/v2/ladi_v2_labels_test_resized.csv"
OUTPUT_CSV = "D:/prediction_output_with_metadata.csv"
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

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle checkpoints saved with "state_dict"
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Fix classifier mismatch if needed
classifier_weight = checkpoint.get("classifier.weight", None)
if classifier_weight is not None and classifier_weight.shape[0] != NUM_CLASSES:
    print(f"⚠️ Adjusting classifier mismatch: checkpoint has {classifier_weight.shape[0]}, expected {NUM_CLASSES}")
    checkpoint.pop("classifier.weight", None)
    checkpoint.pop("classifier.bias", None)

missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"✅ Model loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

model.to(DEVICE)
model.eval()

# ====== Load Metadata ======
metadata_df = pd.read_csv(METADATA_PATH)
print("\n📑 Columns in metadata CSV:", metadata_df.columns.tolist())
print(metadata_df.head(3))

# Try to detect the correct column for image names
possible_name_cols = ["filename", "image_name", "file_name", "img_name"]
name_col = None
for col in metadata_df.columns:
    if col.lower() in possible_name_cols:
        name_col = col
        break
if name_col is None:
    name_col = metadata_df.columns[0]  # fallback: first column
    print(f"⚠️ No standard image name column found, using first column: {name_col}")

metadata_df[name_col] = metadata_df[name_col].astype(str).str.strip()

# ====== Inference Function ======
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predicted_labels = [LABELS[i] for i, p in enumerate(probs) if p > 0.5]
    return probs, predicted_labels

# ====== Run Inference and Merge with Metadata ======
print(f"\n📂 Running inference on images in: {IMAGE_DIR}\n")
results = []

for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(IMAGE_DIR, img_name)
        probs, predicted_labels = predict_image(img_path)

        # Match with metadata
        match = metadata_df[metadata_df[name_col] == img_name]
        if match.empty:
            print(f"⚠️ No metadata for: {img_name}")
            continue

        row = {
            'image_name': img_name,
            'latitude': match.iloc[0].get('latitude', None),
            'longitude': match.iloc[0].get('longitude', None),
            'timestamp': match.iloc[0].get('timestamp', None)
        }

        # Add probabilities
        for i, label in enumerate(LABELS):
            row[label] = round(probs[i], 4)

        row['predicted_labels'] = ", ".join(predicted_labels)
        results.append(row)

# ====== Save to CSV ======
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! Output saved to: {OUTPUT_CSV}")
