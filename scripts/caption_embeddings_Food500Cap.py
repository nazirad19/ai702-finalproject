import torch
import clip
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L/14"
output_path = "/home/Downloads/CV703_FoodGen_Project/food500cap_test_image_embeddings.pt"
batch_size = 32  # Reduce memory pressure

# === Load CLIP model ===
model, preprocess = clip.load(model_name, device=device)
model.eval()

# === Load Dataset ===
dataset = load_dataset("advancedcv/Food500Cap_test")["test"]

# === Collect batched image tensors ===
features, filenames = [], []

for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding in Batches"):
    batch_images = []
    batch_filenames = []

    for j in range(i, min(i + batch_size, len(dataset))):
        img = dataset[j]["image"].convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        batch_images.append(img_tensor)
        batch_filenames.append(f"{j}.jpg")

    batch_images = torch.cat(batch_images).to(device)
    with torch.no_grad():
        batch_feats = model.encode_image(batch_images)
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)

    features.extend(batch_feats.cpu())
    filenames.extend(batch_filenames)

# === Save to file ===
torch.save({
    "features": torch.stack(features),
    "filenames": filenames
}, output_path)

print(f"✅ Saved image embeddings to: {output_path}")
