# import pandas as pd
# import os

# # CSV file containing labels
# labels_csv = r"D:\ladidataset\ladi_v2_resized\v2\ladi_v2_labels_train_full_resized.csv"

# # Root folder where resized images are stored
# images_root = r"D:\ladidataset\ladi_v2_resized\v2\images_resized"

# print("Loading labels...")
# labels_df = pd.read_csv(labels_csv)
# print("Loaded labels:", labels_df.shape)

# # Check a few rows
# print(labels_df.head())

# # Function to fix paths
# def fix_path(p):
#     prefix = os.path.join("v2", "images_resized")
#     # Normalize slashes
#     p = p.replace("/", os.sep)
#     if p.startswith(prefix):
#         # Remove the prefix and leading separator
#         p_fixed = p[len(prefix)+1:]
#     else:
#         p_fixed = p
#     # Join with images_root
#     return os.path.join(images_root, p_fixed)

# # Apply path fix
# labels_df['image_path'] = labels_df['local_path'].apply(fix_path)

# # Check a few constructed paths
# print("\nConstructed image paths:")
# print(labels_df['image_path'].head())

# # Count missing files
# missing = [p for p in labels_df['image_path'] if not os.path.exists(p)]
# print(f"\nMissing images: {len(missing)}")
# if missing:
#     print("Example missing file:", missing[0])
# else:
#     print("All images found!")


# import pandas as pd
# import os

# # Path to your full metadata CSV
# metadata_path = r"D:\ladidataset\image_metadata.csv"

# # Check if the file exists
# if not os.path.exists(metadata_path):
#     raise FileNotFoundError(f"{metadata_path} not found!")

# # Try reading the CSV (handles Excel CSV BOM issues)
# try:
#     metadata = pd.read_csv(metadata_path, encoding='utf-8-sig')
# except pd.errors.EmptyDataError:
#     raise ValueError(f"{metadata_path} is empty or cannot be parsed.")
# except Exception as e:
#     # If CSV reading fails, try reading as Excel
#     try:
#         metadata = pd.read_excel(metadata_path)
#         print("Read file as Excel successfully.")
#     except Exception as e2:
#         raise ValueError(f"Failed to read {metadata_path} as CSV or Excel: {e2}")

# print("Full metadata loaded:", metadata.shape)
# print(metadata.head())

# # Randomly sample 100 rows for testing/deployment
# sampled_metadata = metadata.sample(n=100, random_state=42)  # reproducible sampling

# # Save the sampled rows to a new CSV
# sampled_csv_path = r"D:\ladidataset\image_metadata_sampled_100.csv"
# sampled_metadata.to_csv(sampled_csv_path, index=False)

# print(f"\nSampled 100 rows saved to: {sampled_csv_path}")
# print(sampled_metadata.head())

# *****************_____________________________________________*********************************


# import pandas as pd
# import numpy as np
# import os

# # Paths
# metadata_csv = r"D:\ladidataset\image_metadata.csv"  # full metadata CSV
# sampled_csv = r"D:\ladidataset\image_metadata_sampled_100.csv"  # output for 100-row sample

# # Bounding box for temporary coordinates (example: somewhere in USA)
# # Format: min_lat, max_lat, min_lon, max_lon
# min_lat, max_lat = 37.0, 38.0
# min_lon, max_lon = -122.5, -121.5

# print("Loading full metadata...")
# metadata = pd.read_csv(metadata_csv)
# print(f"Full metadata loaded: {metadata.shape}")

# # Take a random 100-row sample
# sampled = metadata.sample(n=100, random_state=42).reset_index(drop=True)

# # Generate random temporary coordinates
# np.random.seed(42)
# sampled['lat'] = np.random.uniform(min_lat, max_lat, size=sampled.shape[0])
# sampled['lon'] = np.random.uniform(min_lon, max_lon, size=sampled.shape[0])

# # Save the sampled CSV
# sampled.to_csv(sampled_csv, index=False)
# print(f"Sampled metadata with random coordinates saved to: {sampled_csv}")

# # Optional: check first few rows
# print(sampled.head())

# ------------------------------------------------------------------------------#

# import pandas as pd

# sampled_csv = "D:/ladidataset/image_metadata_sampled_100.csv"
# df = pd.read_csv(sampled_csv)
# print(df.head())

# import pandas as pd
# import folium
# import os
# from folium.plugins import MarkerCluster

# # Load the sampled CSV
# csv_path = r"D:\ladidataset\image_metadata_sampled_100.csv"

# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"CSV file not found at {csv_path}")

# labels_df = pd.read_csv(csv_path)

# # Reduce to 25 points
# labels_df = labels_df.sample(n=25, random_state=42).reset_index(drop=True)
# print("CSV loaded successfully. Showing 25 sampled rows:")
# print(labels_df.head())

# # 25 fixed coordinates inside India (lat, lon)
# india_coords = [
#     (28.6139, 77.2090),   # Delhi
#     (19.0760, 72.8777),   # Mumbai
#     (13.0827, 80.2707),   # Chennai
#     (12.9716, 77.5946),   # Bangalore
#     (22.5726, 88.3639),   # Kolkata
#     (26.9124, 75.7873),   # Jaipur
#     (23.0225, 72.5714),   # Ahmedabad
#     (17.3850, 78.4867),   # Hyderabad
#     (15.2993, 74.1240),   # Goa
#     (31.1471, 75.3412),   # Chandigarh
#     (21.1458, 79.0882),   # Nagpur
#     (25.5941, 85.1376),   # Patna
#     (27.1767, 78.0081),   # Agra
#     (26.4499, 80.3319),   # Lucknow
#     (28.4089, 77.3178),   # Noida
#     (29.9457, 78.1642),   # Dehradun
#     (30.7333, 76.7794),   # Chandigarh (repeat if needed)
#     (24.5854, 73.7125),   # Udaipur
#     (20.9517, 85.0985),   # Bhubaneswar
#     (11.0168, 76.9558),   # Coimbatore
#     (9.9312, 76.2673),    # Kochi
#     (16.5062, 80.6480),   # Vijayawada
#     (22.3072, 73.1812),   # Vadodara
#     (18.5204, 73.8567),   # Pune
#     (23.2599, 77.4126)    # Bhopal
# ]

# # Assign each image a fixed India coordinate
# labels_df['lat'] = [coord[0] for coord in india_coords]
# labels_df['lon'] = [coord[1] for coord in india_coords]

# # Create Folium map
# m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
# marker_cluster = MarkerCluster().add_to(m)

# # Add markers with disaster info
# for idx, row in labels_df.iterrows():
#     popup_text = (
#         f"Image: {os.path.basename(row['image_path'])}<br>"
#         f"Water: {row.get('water_any', 'NA')}<br>"
#         f"Flooding: {row.get('flooding_any', 'NA')}<br>"
#         f"Buildings Destroyed: {row.get('buildings_destroyed', 'NA')}<br>"
#         f"Bridges Damage: {row.get('bridges_damage', 'NA')}<br>"
#         f"Trees Damage: {row.get('trees_damage', 'NA')}"
#     )
#     folium.Marker(
#         location=[row['lat'], row['lon']],
#         popup=popup_text
#     ).add_to(marker_cluster)

# # Save the map
# map_save_path = r"D:\GitHub\auto_ai_ops\scripts\image_map_india_25_fixed.html"
# m.save(map_save_path)
# print(f"Map saved successfully! Open {map_save_path} to view it.")




import pandas as pd
import folium
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from folium.plugins import MarkerCluster
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# -----------------------
# Paths
# -----------------------
csv_path = r"D:\ladidataset\image_metadata_sampled_100.csv"
model_path = r"D:\GitHub\auto_ai_ops\best_effnet_b3_multilabel.pth"
map_save_path = r"D:\GitHub\auto_ai_ops\scripts\image_map_india_25_fixed.html"

# -----------------------
# Load CSV and sample 25 images
# -----------------------
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

labels_df = pd.read_csv(csv_path)
labels_df = labels_df.sample(n=25, random_state=42).reset_index(drop=True)
print("CSV loaded and 25 images sampled.")

# -----------------------
# Fixed 25 coordinates strictly inside India
# -----------------------
india_coords = [
    (28.6139, 77.2090), (19.0760, 72.8777), (13.0827, 80.2707), (12.9716, 77.5946),
    (22.5726, 88.3639), (26.9124, 75.7873), (23.0225, 72.5714), (17.3850, 78.4867),
    (15.2993, 74.1240), (31.1471, 75.3412), (21.1458, 79.0882), (25.5941, 85.1376),
    (27.1767, 78.0081), (26.4499, 80.3319), (28.4089, 77.3178), (29.9457, 78.1642),
    (30.7333, 76.7794), (24.5854, 73.7125), (20.9517, 85.0985), (11.0168, 76.9558),
    (9.9312, 76.2673), (16.5062, 80.6480), (22.3072, 73.1812), (18.5204, 73.8567),
    (23.2599, 77.4126)
]
labels_df['lat'] = [coord[0] for coord in india_coords]
labels_df['lon'] = [coord[1] for coord in india_coords]

# -----------------------
# PyTorch Dataset for inference
# -----------------------
class DisasterDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

# -----------------------
# Load trained model
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 15  # matches your saved model

# Recreate model architecture first
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)

# Load state_dict (checkpoint is OrderedDict)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()
print("Model loaded and set to eval mode.")

# -----------------------
# Timm preprocessing
# -----------------------
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

dataset = DisasterDataset(labels_df, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------
# Labels
# -----------------------
all_labels = [
    'water_any', 'flooding_any', 'flooding_structures', 'bridges_any', 'bridges_damage',
    'buildings_any', 'buildings_destroyed', 'buildings_major', 'buildings_minor',
    'roads_any', 'roads_damage', 'trees_any', 'trees_damage', 'other_label1', 'other_label2'
]

display_labels = ['water_any', 'flooding_any', 'buildings_destroyed', 'bridges_damage', 'trees_damage']

# -----------------------
# Run inference and collect predictions
# -----------------------
threshold = 0.3  # lower threshold to catch positive predictions
predictions = []

with torch.no_grad():
    for imgs, img_paths in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        outputs = torch.sigmoid(outputs).cpu()
        preds = (outputs > threshold).int().numpy()[0]

        pred_dict = {}
        for label in display_labels:
            if label in all_labels:
                pred_dict[label] = preds[all_labels.index(label)]
            else:
                pred_dict[label] = 0  # default if missing
        pred_dict['image_path'] = img_paths[0]
        predictions.append(pred_dict)

pred_df = pd.DataFrame(predictions)

# -----------------------
# Merge predictions with coordinates
# -----------------------
map_df = labels_df.merge(pred_df, on='image_path', how='left')
print("Merged DataFrame columns:", map_df.columns)

# -----------------------
# Create Folium map with MarkerCluster
# -----------------------
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

for idx, row in map_df.iterrows():
    popup_text = f"Image: {os.path.basename(row['image_path'])}<br>"
    for label in display_labels:
        popup_text += f"{label.replace('_', ' ').title()}: {row.get(label, 0)}<br>"

    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=popup_text
    ).add_to(marker_cluster)

# Save map
m.save(map_save_path)
print(f"Map saved successfully! Open {map_save_path} to view it.")

