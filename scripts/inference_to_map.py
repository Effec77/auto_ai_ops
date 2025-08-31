# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# import timm

# # ---------------- PARAMETERS -----------------
# CHECKPOINT_PATH = r"D:/GitHub/auto_ai_ops/best_effnet_b3_multilabel.pth"
# IMAGE_FOLDER = r"D:\ladidataset\ladi_v2_resized\v2\images_resized"
# METADATA_CSV = r"D:/ladidataset/image_metadata_sampled_100.csv"
# OUTPUT_CSV = r"D:/GitHub/auto_ai_ops/inference_results.csv"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 4
# NUM_CLASSES = 15  # Corrected number of labels
# IMAGE_SIZE = 224  # Standard EfficientNetB3 input size

# # ---------------- DATASET -----------------
# class DisasterDataset(Dataset):
#     def __init__(self, metadata_csv, image_folder, transform=None):
#         self.df = pd.read_csv(metadata_csv)
#         self.image_folder = Path(image_folder)
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image_path = Path(row['image_path'])
#         if not image_path.is_absolute():
#             image_path = self.image_folder / image_path

#         image = Image.open(image_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, str(image_path)

# # ---------------- TRANSFORMS -----------------
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # ---------------- DATA LOADER -----------------
# dataset = DisasterDataset(METADATA_CSV, IMAGE_FOLDER, transform=transform)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# # ---------------- MODEL -----------------
# model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
# state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)
# model.to(DEVICE)
# model.eval()

# # ---------------- INFERENCE -----------------
# results = []

# with torch.no_grad():
#     for images, paths in tqdm(loader):
#         images = images.to(DEVICE)
#         outputs = model(images)
#         probs = torch.sigmoid(outputs)  # multi-label probabilities
#         probs = probs.cpu().numpy()
#         for path, prob in zip(paths, probs):
#             result = {'image_path': path}
#             result.update({
#                 'bridges_any': prob[0],
#                 'bridges_damage': prob[1],
#                 'buildings_affected': prob[2],
#                 'buildings_any': prob[3],
#                 'buildings_destroyed': prob[4],
#                 'buildings_major': prob[5],
#                 'buildings_minor': prob[6],
#                 'debris_any': prob[7],
#                 'flooding_any': prob[8],
#                 'flooding_structures': prob[9],
#                 'roads_any': prob[10],
#                 'roads_damage': prob[11],
#                 'trees_any': prob[12],
#                 'trees_damage': prob[13],
#                 'water_any': prob[14]
#             })
#             results.append(result)

# # ---------------- SAVE TO CSV -----------------
# results_df = pd.DataFrame(results)
# results_df.to_csv(OUTPUT_CSV, index=False)
# print(f"Inference results saved to {OUTPUT_CSV}")

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import folium
import random

# -------- CONFIG ---------
CHECKPOINT_PATH = r"D:\GitHub\auto_ai_ops\best_effnet_b3_multilabel.pth"
IMAGE_ROOT = r"D:\ladidataset\ladi_v2_resized\v2\images_resized"
METADATA_CSV = r"D:\ladidataset\image_metadata_sampled_100.csv"
OUTPUT_CSV = r"D:\GitHub\auto_ai_ops\sampled_10_predictions.csv"

LABELS = [
    'bridges_any', 'bridges_damage', 'buildings_affected', 'buildings_any',
    'buildings_destroyed', 'buildings_major', 'buildings_minor', 'debris_any',
    'flooding_any', 'flooding_structures', 'roads_any', 'roads_damage',
    'trees_any', 'trees_damage', 'water_any'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- LAND COORDINATES POOL (inside India) ---------
# Predefined lat/lon points strictly on land
LAND_COORDS = [
    (28.6139, 77.2090),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    (13.0827, 80.2707),  # Chennai
    (22.5726, 88.3639),  # Kolkata
    (12.9716, 77.5946),  # Bangalore
    (17.3850, 78.4867),  # Hyderabad
    (26.9124, 75.7873),  # Jaipur
    (23.2599, 77.4126),  # Bhopal
    (21.1458, 79.0882),  # Nagpur
    (25.3176, 82.9739),  # Lucknow
    # add more if needed
]

# -------- LOAD MODEL ---------
import timm
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(LABELS))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- IMAGE TRANSFORM ---------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# -------- LOAD METADATA & SAMPLE 10 ---------
df = pd.read_csv(METADATA_CSV)
df = df.sample(n=10, random_state=42).reset_index(drop=True)

# -------- INFERENCE ---------
predictions = []
for idx, row in df.iterrows():
    img_path = row['image_path']
    image = Image.open(img_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = torch.sigmoid(model(x)).cpu().numpy()[0]
    predictions.append([img_path] + list(y))

pred_df = pd.DataFrame(predictions, columns=['image_path'] + LABELS)
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")

# -------- ASSIGN LAND COORDS & CREATE MAP ---------
map_points = []
for idx, row in pred_df.iterrows():
    lat, lon = LAND_COORDS[idx]  # take one land point from pool
    disasters = [label for label in LABELS if row[label] >= 0.5]
    popup_text = ', '.join(disasters) if disasters else 'No major disaster'
    map_points.append({'lat': lat, 'lon': lon, 'popup': popup_text})

m = folium.Map(location=[22, 80], zoom_start=5)
for pt in map_points:
    folium.Marker([pt['lat'], pt['lon']], popup=pt['popup']).add_to(m)

MAP_OUTPUT = r"D:\GitHub\auto_ai_ops\india_disaster_map.html"
m.save(MAP_OUTPUT)
print(f"Map saved to {MAP_OUTPUT}")
