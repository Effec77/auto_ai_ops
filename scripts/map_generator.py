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
# Fixed 25 coordinates inside India
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
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
model.to(device)

# Load state_dict
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
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
threshold = 0.3  # lower threshold to capture more positives
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
                pred_dict[label] = 0  # default to 0 if label missing
        pred_dict['image_path'] = img_paths[0]
        predictions.append(pred_dict)

pred_df = pd.DataFrame(predictions)

# -----------------------
# Merge predictions with coordinates safely
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
