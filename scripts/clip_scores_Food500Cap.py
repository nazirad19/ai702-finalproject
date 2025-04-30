import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# === CONFIG ===
output_scores_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_3_FoodCap_6decoder_subset/clip_scores_food500cap_test.json"
captions_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_3_FoodCap_6decoder_subset/captions_finetuned_food500_test.json"  # generated captions
batch_size = 32
model_name = "ViT-L/14"
threshold = 0.28

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Loading CLIP model: {model_name}")
model, preprocess = clip.load(model_name, device=device)
model.eval()

# === Load captions ===
with open(captions_file, "r") as f:
    captions_dict = json.load(f)

# === Load HuggingFace Test Set ===
print("📦 Loading Food500Cap test set from HuggingFace...")
hf_dataset = load_dataset("advancedcv/Food500Cap_test")["test"]
print(f"✅ Loaded {len(hf_dataset)} test images.")

# === Custom Dataset ===
class CaptionedImageDataset(Dataset):
    def __init__(self, hf_dataset, captions_dict):
        self.samples = []
        for idx, item in enumerate(hf_dataset):
            fname = f"{idx}.jpg"
            if fname in captions_dict:
                self.samples.append((item["image"], captions_dict[fname], fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, caption, fname = self.samples[idx]
        try:
            img = preprocess(img.convert("RGB"))
            return img, caption, fname
        except Exception:
            return None, caption, fname

# === Collate Function to Skip Corrupt ===
def collate_skip_none(batch):
    filtered = [b for b in batch if b[0] is not None]
    if not filtered:
        return [], [], []
    imgs, caps, fnames = zip(*filtered)
    return imgs, caps, fnames

# === Build DataLoader ===
dataset = CaptionedImageDataset(hf_dataset, captions_dict)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_none)

# === Scoring ===
scores = {}
failures = []

print(f"📊 Scoring {len(dataset)} image-caption pairs...")
for images, captions, filenames in tqdm(dataloader, desc="Scoring Batches"):
    if not images:
        failures.extend(filenames)
        continue

    images = torch.stack(images).to(device)
    try:
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(clip.tokenize(captions).to(device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).diag()

        for fname, caption, score in zip(filenames, captions, similarities):
            scores[fname] = {
                "caption": caption,
                "clip_score": score.item()
            }

    except Exception as e:
        failures.extend(filenames)
        print(f"❌ Failed scoring batch: {e}")

# === Save Scores ===
sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]["clip_score"], reverse=True))
with open(output_scores_file, "w") as f:
    json.dump(sorted_scores, f, indent=2)

# === Summary ===
above_threshold = sum(1 for x in scores.values() if x["clip_score"] > threshold)
print(f"✅ Scoring complete. Saved to: {output_scores_file}")
print(f"📈 Total scored: {len(scores)}")
print(f"🎯 CLIP score > {threshold}: {above_threshold}")
print(f"⚠️ Failed to process {len(failures)} files.")
