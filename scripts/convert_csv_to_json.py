import pandas as pd
import json
import ast

# Load CSV
csv_path = "D:/prediction_output_with_metadata.csv"
df = pd.read_csv(csv_path)

# Optional: rename columns if needed
# df.rename(columns={'img_id': 'image_name'}, inplace=True)

# Clean & convert predicted_labels into list
def parse_labels(label_str):
    try:
        # Try evaluating it as a Python list (if in format: "['flooding', 'fire']")
        return ast.literal_eval(label_str)
    except:
        # Otherwise split by comma
        return [label.strip() for label in str(label_str).split(',')]

df['predicted_labels'] = df['predicted_labels'].apply(parse_labels)

# Select only required fields
json_data = []
for _, row in df.iterrows():
    json_data.append({
        "image_name": row["image_name"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "timestamp": row.get("timestamp", ""),
        "predicted_labels": row["predicted_labels"]
    })

# Save to JSON
output_path = "D:/map_data.json"
with open(output_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ JSON exported to: {output_path}")
