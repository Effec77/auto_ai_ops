import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# ========= Paths =========
csv_path = "D:/images/v2/ladi_v2_labels_train_full_resized.csv"
images_root = "D:/images"

# ========= Read CSV =========
df = pd.read_csv(csv_path)

# ✅ Set the correct column for image paths
image_col = 'local_path'

# Drop rows with missing image paths
df = df[df[image_col].notna()]

# Build full absolute image paths
df['full_path'] = [
    os.path.join(images_root, str(x).replace('/', os.sep))
    for x in df[image_col]
]

# Filter out rows with missing image files
# Build full absolute image paths using list comprehension
df['full_path'] = [
    os.path.join(images_root, str(x).replace('/', os.sep))
    for x in df['local_path'].tolist()
]

# Print sample full paths
print("🔍 Sample full paths:")
for path in df['full_path'][:10]:
    print(path)

# Check how many files actually exist
exist_flags = [os.path.exists(p) for p in df['full_path']]
print(f"✅ Found {sum(exist_flags)} out of {len(df)} valid images")

# Keep only valid paths
df = df.loc[exist_flags].reset_index(drop=True)

# Abort if still empty
if df.empty:
    raise ValueError("❌ No valid image paths found! Check your 'images_root' and 'local_path' values.")


# ✅ Generate synthetic labels (since original doesn't have a 'category' column)
df['category'] = [os.path.basename(os.path.dirname(x)) for x in df['full_path']]

# ========= Encode labels =========
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['category'])

# Drop classes with only one image (required for stratified split)
label_counts = df['label_encoded'].value_counts()
df = df[df['label_encoded'].isin(label_counts[label_counts > 1].index)]


# ========= Train-Validation Split =========
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label_encoded'],
    random_state=42
)

# ========= PyTorch Dataset Class =========
class LadiDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, 'full_path']
        label = self.data.loc[idx, 'label_encoded']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ========= Transforms =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========= Dataset & DataLoader =========
train_dataset = LadiDataset(train_df, transform=transform)
val_dataset = LadiDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"✅ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
