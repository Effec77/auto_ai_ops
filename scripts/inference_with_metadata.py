import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ====== Config ======
MODEL_PATH = "D:/Auto_aiops/checkpoints/best_effnet_b3_multilabel.pth"
IMAGE_DIR = "D:/inference_images"
METADATA_PATH = "D:/image_metadata.csv"
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

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ====== Load Model ======
print(f"\n🔁 Loading model from: {MODEL_PATH}")
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded")

# ====== Load Metadata ======
metadata_df = pd.read_csv(METADATA_PATH)
metadata_df['image_name'] = metadata_df['image_name'].astype(str).str.strip()

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

        # Exact match with full filename
        match = metadata_df[metadata_df['image_name'] == img_name]
        if match.empty:
            print(f"⚠️ No metadata for: {img_name}")
            continue

        lat = match.iloc[0]['latitude']
        lon = match.iloc[0]['longitude']
        ts = match.iloc[0]['timestamp']

        row = {
            'image_name': img_name,
            'latitude': lat,
            'longitude': lon,
            'timestamp': ts
        }

        # Add probabilities to the row
        for i, label in enumerate(LABELS):
            row[label] = round(probs[i], 4)

        row['predicted_labels'] = ", ".join(predicted_labels)
        results.append(row)

# ====== Save to CSV ======
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done! Output saved to: {OUTPUT_CSV}")
