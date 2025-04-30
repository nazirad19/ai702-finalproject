import json
import torch
import clip
from tqdm import tqdm

# === CONFIG ===
captions_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned.json"

output_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_embeddings.pt"
model_name = "ViT-L/14"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _ = clip.load(model_name, device=device)
model.eval()

with open(captions_file, "r") as f:
    data = json.load(f)

filenames = list(data.keys())
captions = list(data.values())

print(f"💬 Loaded {len(captions)} captions...")

features = []
batched_filenames = []

# === Encode in batches ===
for i in tqdm(range(0, len(captions), batch_size), desc="🔁 Encoding"):
    batch_captions = captions[i:i + batch_size]
    batch_filenames = filenames[i:i + batch_size]
    
    # Tokenize and truncate to 77 tokens safely
    try:
        tokens = clip.tokenize(batch_captions, truncate=True).to(device)
    except Exception as e:
        print(f"❌ Error tokenizing batch {i}: {e}")
        continue
    
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu())
        batched_filenames.extend(batch_filenames)

# === Save ===
final_feats = torch.cat(features)
torch.save({"features": final_feats, "filenames": batched_filenames}, output_file)
print(f"✅ Saved caption embeddings to {output_file}")
