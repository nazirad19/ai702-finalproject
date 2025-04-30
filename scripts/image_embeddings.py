import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
image_root = "/home/Downloads/CV703_Project/food-101/food-101/food-101/images"  # <-- update this
image_list_file = "synced_fixed_image_list.txt"
output_file = "image_embeddings.pt"
model_name = "ViT-L/14"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load CLIP model ===
print(f"🔍 Loading CLIP model: {model_name}")
model, preprocess = clip.load(model_name, device=device)
model.eval()

# === Read image filenames from txt ===
with open(image_list_file, "r") as f:
    wanted_filenames = set(line.strip() for line in f if line.strip())

print(f"📄 Found {len(wanted_filenames)} image names in list.")

# === Collect only wanted image paths ===
image_paths = []
for root, _, files in os.walk(image_root):
    for file in files:
        if file in wanted_filenames:
            image_paths.append(os.path.join(root, file))

print(f"✅ Matched {len(image_paths)} images in dataset.")

# === Encode images ===
features = []
filenames = []

for path in tqdm(image_paths, desc="🚀 Encoding Images"):
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu())
        filenames.append(os.path.basename(path))
    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")

print(filenames)
# === Save embeddings ===
image_features = torch.cat(features)
torch.save({"features": image_features, "filenames": filenames}, output_file)
print(f"💾 Saved {len(filenames)} image embeddings to {output_file}")
