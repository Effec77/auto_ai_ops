# scripts/convert_predictions_to_geojson.py
import os
import json
import pandas as pd

# ---------- CONFIG - change these if you want ----------
PREDICTIONS_CSV = r"D:/GitHub/auto_ai_ops/csvResults/predictions_with_coords.csv"
OUTPUT_GEOJSON = r"D:/GitHub/auto_ai_ops/csvResults/predictions_map.geojson"

# Disaster labels in the same order you use in inference CSV
DISASTER_LABELS = [
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
    "water_any",
]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def main():
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(f"Predictions CSV not found: {PREDICTIONS_CSV}")

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded {len(df)} rows from {PREDICTIONS_CSV}")
    # validate presence of lat/lon
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV must contain 'lat' and 'lon' columns.")

    features = []
    skipped = 0
    for idx, row in df.iterrows():
        try:
            lat = safe_float(row["lat"])
            lon = safe_float(row["lon"])
        except Exception:
            skipped += 1
            continue

        # skip obviously invalid coordinates
        if pd.isna(lat) or pd.isna(lon):
            skipped += 1
            continue

        # build properties: include all disaster labels that exist in the CSV
        properties = {}
        for label in DISASTER_LABELS:
            if label in row:
                properties[label] = safe_float(row[label])

        # compute top label and score (over the labels we have)
        if properties:
            # get label-value pairs
            label_items = list(properties.items())
            # pick the label with maximum score
            top_label, top_score = max(label_items, key=lambda kv: kv[1])
        else:
            top_label, top_score = None, 0.0

        # include image_path if present
        if "image_path" in row:
            properties["image_path"] = row["image_path"]

        # add top_label/top_score fields (use None for top_label if no labels)
        properties["top_label"] = top_label if top_label is not None else ""
        properties["top_score"] = float(top_score)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": properties
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"✅ GeoJSON saved to: {OUTPUT_GEOJSON}")
    if skipped:
        print(f"⚠️ Skipped {skipped} rows due to missing/invalid coords")

if __name__ == "__main__":
    main()
