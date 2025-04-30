import os
import json
import torch
import clip
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
hf_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# === CONFIG ===
image_root = "/home/Downloads/CV703_FoodGen_Project/food-101/food-101/food-101/images"
captions_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned.json"
output_scores_file = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_clip_scores.json"
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

# === Load previous scores if available ===
existing_scores = {}
if os.path.exists(output_scores_file):
    with open(output_scores_file, "r") as f:
        existing_scores = json.load(f)
    print(f"♻️ Loaded {len(existing_scores)} previously scored items.")

already_scored = set(existing_scores.keys())

# === Index image files ===
print("🔍 Indexing image files...")
filename_to_path = {
    file: os.path.join(root, file)
    for root, _, files in os.walk(image_root)
    for file in files
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
}
print(f"✅ Indexed {len(filename_to_path)} image files.")

# === Custom Dataset ===
class CaptionedImageDataset(Dataset):
    def __init__(self, captions_dict, already_done, filename_to_path):
        self.items = [
            (filename_to_path[f], cap, f)
            for f, cap in captions_dict.items()
            if f not in already_done and f in filename_to_path
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, caption, fname = self.items[idx]
        try:
            img = preprocess(Image.open(path).convert("RGB"))
            return img, caption, fname
        except UnidentifiedImageError:
            return None, caption, fname

# === Build Dataloader ===
print("🧱 Building dataset...")
dataset = CaptionedImageDataset(captions_dict, already_scored, filename_to_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f"📊 Scoring {len(dataset)} image-caption pairs...")

# === Scoring Loop ===
failures = []


for batch in tqdm(dataloader, desc="Scoring Batches"):
    images, captions, filenames = batch

    # Skip failed images
    good_indices = [i for i, img in enumerate(images) if img is not None]
    if not good_indices:
        failures.extend(filenames)
        continue

    images = torch.stack([images[i] for i in good_indices]).to(device)
    captions = [captions[i] for i in good_indices]
    filenames = [filenames[i] for i in good_indices]

    try:
        with torch.no_grad():
            image_features = model.encode_image(images)
            # Truncate text before passing to clip.tokenize
            def safe_truncate_text(caption, max_tokens=77):
                tokens = hf_tokenizer.encode(caption, truncation=True, max_length=max_tokens, add_special_tokens=True)
                return hf_tokenizer.decode(tokens, skip_special_tokens=True)

            truncated_captions = [safe_truncate_text(c) for c in captions]
            text_inputs = clip.tokenize(truncated_captions).to(device)
            text_features = model.encode_text(text_inputs)


            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).diag()

        for fname, caption, score in zip(filenames, captions, similarities):
            existing_scores[fname] = {
                "caption": caption,
                "clip_score": score.item()
            }

    except Exception as e:
        failures.extend(filenames)
        print(f"❌ Failed scoring batch: {e}")

# === Save scores (sorted)
sorted_scores = dict(sorted(
    existing_scores.items(),
    key=lambda item: item[1]["clip_score"],
    reverse=True
))
with open(output_scores_file, "w") as f:
    json.dump(sorted_scores, f, indent=2)

# === Summary
above_threshold = sum(1 for x in existing_scores.values() if x["clip_score"] > threshold)
print(f"✅ Scoring complete. Saved to: {output_scores_file}")
print(f"📈 Total scored: {len(existing_scores)}")
print(f"🎯 CLIP score > {threshold}: {above_threshold}")
print(f"⚠️ Failed to process {len(failures)} files.")
