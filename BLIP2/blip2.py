import os
import json
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time
import csv
from collections import Counter, defaultdict
import re

# === CONFIG ===
data_root = "/home/karina.abubakirova/.cache/kagglehub/datasets/dansbecker/food-101/versions/1/food-101/food-101/images"
output_path = "food_images_captions_blip2.json"
csv_output_path = "food_images_captions_blip2.csv"
model_name = "Salesforce/blip2-flan-t5-xl"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
list_file = "synced_fixed_image_list.txt"

# === Load Model and Processor ===
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Dataset that finds images by filename only ===
class FoodDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.filenames = []

        with open(list_file, "r") as f:
            base_filenames = [line.strip() for line in f if line.strip()]

        # Build a mapping from base filename to full path
        base_to_path = {}
        duplicates = set()
        for img_path in self.root_dir.rglob("*.*"):
            if img_path.name in base_filenames:
                if img_path.name not in base_to_path:
                    base_to_path[img_path.name] = img_path
                else:
                    # Duplicate filename
                    duplicates.add(img_path.name)

        if duplicates:
            print(f"⚠️ WARNING: Duplicate base filenames found: {sorted(duplicates)[:10]}... ({len(duplicates)} total)")

        for fname in base_filenames:
            if fname in base_to_path:
                img_path = base_to_path[fname]
                self.image_paths.append(img_path)
                self.labels.append(img_path.parent.name)
                self.filenames.append(fname)

        print(f"✅ Found {len(self.image_paths)} valid images out of {len(base_filenames)} requested in list.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        filename = self.filenames[idx]  # base filename only
        if self.transform:
            image = self.transform(image)
        return image, label, filename

# === Image Transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Clean-up Helpers ===
unwanted_keywords = ["person", "people", "laptop", "computer", "screen", "television", "monitor"]

def is_caption_clean(caption):
    return not any(word in caption.lower() for word in unwanted_keywords)

def cleanup_caption(caption):
    caption = caption.strip().replace("..", ".").replace(" ,", ",")
    if not caption.endswith("."):
        caption += "."
    return caption[0].upper() + caption[1:]

def smooth_caption(caption):
    if "," in caption and not any(word in caption.lower() for word in ["and", "or"]):
        parts = [p.strip() for p in caption.split(",")]
        if len(parts) > 1:
            return ", ".join(parts[:-1]) + " and " + parts[-1] + "."
    return caption

def remove_redundant_phrases(caption):
    return re.sub(r"\b(\w+)\s+(in|on|with)\s+(a\s+)?\1\b", r"\1", caption, flags=re.IGNORECASE)

def generate_captions(images):
    prompt = "Give a short, concise description of this food image in one sentence. Do not add extra details or guess unobservable context."
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=75,
            min_length=1,
            top_p=0.7,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1.0,
        )

    raw_captions = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [remove_redundant_phrases(smooth_caption(cleanup_caption(c))) for c in raw_captions]

# === Dataset Setup ===
dataset = FoodDataset(data_root, list_file, transform=transform)
print(f"📊 Total selected images: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

captions_dict = {}
csv_rows = []
skipped_counts = Counter()
start_time = time.time()

# === Captioning Loop ===
for images, labels, filenames in dataloader:
    images_pil = [transforms.ToPILImage()(img).convert("RGB") for img in images]
    try:
        raw_captions = generate_captions(images_pil)
        for fname, caption, label in zip(filenames, raw_captions, labels):
            if is_caption_clean(caption):
                final_caption = f"{caption} (label: {label})"
                captions_dict[fname] = final_caption
                csv_rows.append([fname, final_caption])
                print(f"[{len(captions_dict)}] {fname}: {final_caption}")
            else:
                skipped_counts[label] += 1
                print(f"⚠️ Skipped non-food caption for {fname} (label: {label}): {caption}")

        if len(captions_dict) % 1000 == 0:
            with open("checkpoint_blip2.json", "w") as f:
                json.dump(captions_dict, f, indent=4)
            print("💾 Intermediate checkpoint saved.")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error generating caption for batch {filenames}: {e}")
        continue

# === Save Outputs ===
with open(output_path, "w") as f:
    json.dump(captions_dict, f, indent=4)

with open(csv_output_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "caption"])
    writer.writerows(csv_rows)

print(f"✅ Captions saved to: {output_path} and {csv_output_path} | Total: {len(captions_dict)}")
print(f"⏱️ Elapsed time: {time.time() - start_time:.2f}s")

print("🔍 Skipped captions per class:")
for label, count in skipped_counts.items():
    print(f"{label}: {count}")
