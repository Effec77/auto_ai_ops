import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import timm

def get_model():
    # EfficientNet-B3 with 15 output classes (adjust if needed)
    return timm.create_model('efficientnet_b3', pretrained=False, num_classes=15)

# === Paths ===
MODEL_PATH = "D:\\GitHub\\auto_ai_ops\\checkpoints\\best_effnet_b3_multilabel.pth"
IMAGE_METADATA = "D:\\ladidataset\\image_metadata_sampled_100.csv"  # must have: image_path, lat, lon
OUTPUT_CSV = "D:/GitHub/auto_ai_ops/csvResults/predictions_with_coords.csv"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# === Load Metadata (provides lat/lon per image) ===
metadata_df = pd.read_csv(IMAGE_METADATA)

# Disaster class labels (15 total)
CLASS_LABELS = [
    "bridges_any",
    "bridges_damage",
    "buildings_affected",
    "buildings_any",
    "buildings_destroyed",
    "buildings_major",
    "buildings_minor",
    "debris_any",
    "flooding_any",
    "flooding_structures",
    "roads_any",
    "roads_damage",
    "trees_any",
    "trees_damage",
    "water_any"
]

# === Run Inference ===
results = []
for _, row in metadata_df.iterrows():
    img_path = row["image_path"]
    lat, lon = row.get("lat"), row.get("lon")

    if not os.path.exists(img_path):
        print(f"⚠️ Missing file: {img_path}")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(image)).cpu().numpy().flatten()

        # Save results row with actual class labels
        results.append({
            "image_path": img_path,
            "lat": lat,
            "lon": lon,
            **{CLASS_LABELS[i]: float(preds[i]) for i in range(len(CLASS_LABELS))}
        })

    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")

# === Save Predictions with Coordinates ===
if results:
    pred_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pred_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions with coordinates saved to {OUTPUT_CSV}")
else:
    print("❌ No predictions generated.")
