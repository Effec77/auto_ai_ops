import os
import random
import torch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from torchvision import transforms
from PIL import Image
import timm
import json

# === Paths ===
MODEL_PATH = "D:\\GitHub\\auto_ai_ops\\checkpoints\\best_effnet_b3_multilabel.pth"
IMAGE_METADATA = "D:\\ladidataset\\image_metadata_sampled_100.csv"  # must have: image_path, lat, lon
INDIA_BOUNDARY_GEOJSON = "D:\\GitHub\\auto_ai_ops\\csvResults\\india_coords.geojson"
OUTPUT_GEOJSON = "D:\\GitHub\\auto_ai_ops\\csvResults\\predictions_map.geojson"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
def get_model():
    return timm.create_model('efficientnet_b3', pretrained=False, num_classes=15)

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# === Load Metadata ===
metadata_df = pd.read_csv(IMAGE_METADATA)
metadata_df = metadata_df.sample(n=10, random_state=42)  # select 10 images

# === Load India Polygon ===
india_boundary = gpd.read_file(INDIA_BOUNDARY_GEOJSON)
india_polygon = india_boundary.unary_union  # single Polygon/MultiPolygon

def random_point_in_india():
    minx, miny, maxx, maxy = india_polygon.bounds
    while True:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if india_polygon.contains(point):
            return point.y, point.x  # lat, lon

# === Run Inference and Generate GeoJSON Features ===
features = []

for _, row in metadata_df.iterrows():
    img_path = row["image_path"]
    
    if not os.path.exists(img_path):
        print(f"⚠️ Missing file: {img_path}")
        continue

    try:
        # Run image inference
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(image)).cpu().numpy().flatten()

        # Get a random India-land coordinate
        lat, lon = random_point_in_india()

        # Prepare properties with proper labels
        properties = {
            "image_path": img_path,
            "lat": lat,
            "lon": lon,
            "bridges_any": float(preds[0]),
            "bridges_damage": float(preds[1]),
            "buildings_affected": float(preds[2]),
            "buildings_any": float(preds[3]),
            "buildings_destroyed": float(preds[4]),
            "buildings_major": float(preds[5]),
            "buildings_minor": float(preds[6]),
            "debris_any": float(preds[7]),
            "flooding_any": float(preds[8]),
            "flooding_structures": float(preds[9]),
            "roads_any": float(preds[10]),
            "roads_damage": float(preds[11]),
            "trees_any": float(preds[12]),
            "trees_damage": float(preds[13]),
            "water_any": float(preds[14])
        }

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": properties
        }
        features.append(feature)

    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")

# === Save GeoJSON ===
geojson_data = {"type": "FeatureCollection", "features": features, "name": "predictions_map"}
os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(geojson_data, f, indent=2)

print(f"✅ Predictions GeoJSON saved to {OUTPUT_GEOJSON}")
