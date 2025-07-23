# Upgraded script with EfficientNet-B3, batch size = 16, data augmentation,
# label smoothing, and optimizer/scheduler enhancements

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import timm
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# ========== Config ==========
csv_path = "D:/images/v2/ladi_v2_labels_train_full_resized.csv"
images_root = "D:/images"

BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Data ==========
df = pd.read_csv(csv_path)
df = df[df['local_path'].notna()].copy()
df['full_path'] = [os.path.join(images_root, str(x).replace("/", os.sep)) for x in df['local_path']]
df = df[df['full_path'].apply(os.path.exists)].copy()

# Detect multi-label columns
metadata_cols = {'url', 'local_path'}
label_columns = [col for col in df.columns if col not in metadata_cols and df[col].dropna().isin([0, 1]).all()]
print(f"\U0001F4CC Multi-label columns: {label_columns}")
num_classes = len(label_columns)

# Train-val split
# Fallback: standard split (still acceptable for large datasets)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# ========== Dataset ==========
class LadiDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'full_path']
        labels = self.df.loc[idx, label_columns].values.astype(np.float32)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

# ========== Transforms ==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_loader = DataLoader(LadiDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(LadiDataset(val_df, val_transform), batch_size=BATCH_SIZE, pin_memory=True)

# ========== Model ==========
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
model.to(DEVICE)

# ========== Loss, Optimizer, Scheduler ==========
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_f1 = 0.0

# ========== Training Loop ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"\✅ Epoch {epoch+1}: Train Loss = {total_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    binarized_preds = (all_preds > 0.5).astype(int)

    f1 = f1_score(all_labels, binarized_preds, average='micro', zero_division=0)
    precision = precision_score(all_labels, binarized_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, binarized_preds, average='micro', zero_division=0)
    print(f"\📊 F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_effnet_b3_multilabel.pth")
        print("📀 Model saved (Best F1 so far)")

    scheduler.step()

print("✅ Training complete.")
