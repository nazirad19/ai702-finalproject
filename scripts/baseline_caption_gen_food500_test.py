import os
import json
import torch
import time
import csv
import random
import re
from pathlib import Path
from collections import Counter
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from datasets import load_dataset

# === CONFIG ===
output_json = "captions_baseline_food500_test.json"
output_csv = "captions_baseline_food500_test.csv"
model_path = "Salesforce/instructblip-flan-t5-xl"
checkpoint_json = "captions_baseline_checkpoint_food500_test.json"
checkpoint_csv = "captions_baseline_checkpoint_food500_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8

# === Load Model and Processor ===
processor = InstructBlipProcessor.from_pretrained(model_path)
model = InstructBlipForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# === Load Dataset ===
print("📦 Loading Food500Cap TEST split...")
dataset = load_dataset("advancedcv/Food500Cap_test")["test"]

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Dataset ===
class FoodDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform
        
        print(f"✅ Loaded {len(self.data)} images from HuggingFace dataset.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        label = item["cat"]
        filename = f"{idx}.jpg"

        # ✨ Force RGB conversion
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, filename

    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     image = item["image"]
    #     label = item["cat"]
    #     filename = f"{idx}.jpg"

    #     # filename = f"{item['img_id']}.jpg" if "id" in item else f"{idx}.jpg"

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, label, filename

# === Clean-up Utilities ===
unwanted = ["person", "people", "computer", "screen", "monitor", "television"]

def is_food_caption(text):
    return not any(word in text.lower() for word in unwanted)

def cleanup_caption(text):
    text = text.strip().replace("..", ".").replace(" ,", ",")
    if not text.endswith("."):
        text += "."
    return text[0].upper() + text[1:]

def smooth_caption(text):
    if "," in text and not any(x in text.lower() for x in ["and", "or"]):
        parts = [p.strip() for p in text.split(",")]
        if len(parts) > 1:
            return ", ".join(parts[:-1]) + " and " + parts[-1] + "."
    return text

def remove_redundancy(text):
    return re.sub(r"\b(\w+)\s+(in|on|with)\s+(a\s+)?\1\b", r"\1", text, flags=re.IGNORECASE)

def final_caption_cleanup(text):
    return remove_redundancy(smooth_caption(cleanup_caption(text)))

# === Caption Generation ===
def generate_captions(images):
    prompt = "Give a short, concise description of this food image in one sentence. Do not add extra details or guess unobservable context."
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=75,
            top_p=0.7,
            repetition_penalty=1.5,
            length_penalty=1.0
        )

    raw = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [final_caption_cleanup(cap) for cap in raw]

# === Main Pipeline ===
dataset = FoodDataset(dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# === Resume from checkpoint ===
captions_dict, csv_rows = {}, []
if os.path.exists(checkpoint_json):
    print("📦 Resuming from checkpoint...")
    with open(checkpoint_json, "r") as f:
        captions_dict = json.load(f)
    if os.path.exists(checkpoint_csv):
        with open(checkpoint_csv, "r") as f:
            next(f)
            csv_rows = [row.strip().split(",", 1) for row in f.readlines()]
processed_set = set(captions_dict.keys())

skipped = Counter()
start = time.time()

for images, labels, filenames in dataloader:
    pil_images = [transforms.ToPILImage()(img).convert("RGB") for img in images]
    skip_mask = [name in processed_set for name in filenames]

    if all(skip_mask):
        continue

    try:
        captions = generate_captions(pil_images)
        for name, caption, label, skip in zip(filenames, captions, labels, skip_mask):
            if skip:
                continue
            if is_food_caption(caption):
                final = f"{caption} (label: {label})"
                captions_dict[name] = final
                csv_rows.append([name, final])
                print(f"[{len(captions_dict)}] {name}: {final}")
                with open("caption_log.txt", "a") as log_file:
                    log_file.write(f"{name}: {final}\n")
            else:
                skipped[label] += 1
                print(f"⚠️ Skipped {name}: {caption}")

        if len(captions_dict) % 1000 == 0:
            with open(checkpoint_json, "w") as f:
                json.dump(captions_dict, f, indent=2)
            with open(checkpoint_csv, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "caption"])
                writer.writerows(csv_rows)
            print("💾 Saved intermediate checkpoint.")

        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error with batch {filenames}: {e}")
        continue

# === Save Final Outputs ===
with open(output_json, "w") as f_json:
    json.dump(captions_dict, f_json, indent=4)

with open(output_csv, "w", newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["filename", "caption"])
    writer.writerows(csv_rows)

if os.path.exists(checkpoint_json): os.remove(checkpoint_json)
if os.path.exists(checkpoint_csv): os.remove(checkpoint_csv)

print(f"✅ Captions saved to: {output_json} and {output_csv}")
print(f"⏱️ Elapsed: {time.time() - start:.2f}s")
print("🔍 Skipped captions by class:")
for cls, count in skipped.items():
    print(f"{cls}: {count}")
