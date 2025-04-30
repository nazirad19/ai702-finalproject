import os
import json
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import csv
import random
from collections import Counter
import re

# === CONFIG ===
data_root = "/home/karina.abubakirova/.cache/kagglehub/datasets/dansbecker/food-101/versions/1/food-101/food-101/images"
output_path = "food_images_captions_blip_new.json"
csv_output_path = "food_images_captions_fixed.csv"
model_name = "Salesforce/blip-image-captioning-base"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_images = 20000
images_per_class = max_images // 101
list_file = "fixed_image_list.txt"

# === Load Model and Processor ===
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Custom Dataset for Food Images ===
class FoodDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.filenames = []

        # Load filenames
        with open(list_file, "r") as f:
            target_filenames = set(line.strip() for line in f if line.strip())

        # Walk entire dataset tree
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file in target_filenames and file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, file)
                    label = Path(root).name
                    self.image_paths.append(full_path)
                    self.labels.append(label)
                    self.filenames.append(file)

        print(f"✅ Found {len(self.image_paths)} out of {len(target_filenames)} requested images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        label = self.labels[idx]
        filename = self.filenames[idx]
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
    lower = caption.lower()
    return not any(word in lower for word in unwanted_keywords)

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
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=75,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )

    raw_captions = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [remove_redundant_phrases(smooth_caption(cleanup_caption(c))) for c in raw_captions]

# === Dataset Setup ===
dataset = FoodDataset(data_root, list_file, transform=transform)
print(f"📊 Total selected diverse images: {len(dataset)}")

# with open(list_file, "w") as f:
#     for fname in dataset.filenames:
#         f.write(fname + "\n")

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
                print(f"[{len(captions_dict)}/{max_images}] {fname}: {final_caption}")
            else:
                skipped_counts[label] += 1
                print(f"⚠️ Skipped non-food caption for {fname} (label: {label}): {caption}")

        if len(captions_dict) % 1000 == 0:
            with open("checkpoint.json", "w") as f:
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
