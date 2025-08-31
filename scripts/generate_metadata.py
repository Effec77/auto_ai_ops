import pandas as pd
import os
from PIL import Image  # for reading image dimensions

# CSV containing labels
labels_csv = r"D:\ladidataset\ladi_v2_resized\v2\ladi_v2_labels_train_full_resized.csv"

# Root folder where images are stored
images_root = r"D:\ladidataset\ladi_v2_resized\v2\images_resized"

# Output metadata CSV
metadata_csv = r"D:\ladidataset\image_metadata.csv"

print("Loading labels...")
labels_df = pd.read_csv(labels_csv)
print("Loaded labels:", labels_df.shape)

# Fix image paths (same as check_labels)
def fix_path(p):
    prefix = os.path.join("v2", "images_resized")
    p = p.replace("/", os.sep)
    if p.startswith(prefix):
        p_fixed = p[len(prefix)+1:]
    else:
        p_fixed = p
    return os.path.join(images_root, p_fixed)

labels_df['image_path'] = labels_df['local_path'].apply(fix_path)

# Prepare list to store metadata
metadata_list = []

print("Generating metadata...")
for idx, row in labels_df.iterrows():
    path = row['image_path']
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                width, height = img.size
            # Collect metadata
            metadata = {
                'image_path': path,
                'width': width,
                'height': height,
                **{col: row[col] for col in labels_df.columns if col not in ['local_path', 'image_path']}
            }
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    else:
        print(f"Missing image (skipped): {path}")

# Create metadata DataFrame
metadata_df = pd.DataFrame(metadata_list)
print("Metadata generated:", metadata_df.shape)

# Save to CSV
metadata_df.to_csv(metadata_csv, index=False)
print(f"Metadata saved to {metadata_csv}")
