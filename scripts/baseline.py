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

# === CONFIG ===
data_root = "/home/Downloads/CV703_Project/food-101/food-101/food-101/images"
output_json = "food_images_captions_fixed.json"
output_csv = "food_images_captions_fixed.csv"
fixed_list_file = "fixed_image_list.txt"
model_name = "Salesforce/instructblip-flan-t5-xl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_images = 100
images_per_class = max(1, max_images // 101)


# === Load Model and Processor ===
processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Dataset ===
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels, self.filenames = [], [], []

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                images = [f for f in sorted(os.listdir(label_path)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                selected = random.sample(images, min(images_per_class, len(images)))
                for img in selected:
                    self.image_paths.append(os.path.join(label_path, img))
                    self.labels.append(label)
                    self.filenames.append(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.filenames[idx]

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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
dataset = FoodDataset(data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Adjust images_per_class or check image path.")


with open(fixed_list_file, "w") as f:
    for name in dataset.filenames:
        f.write(name + "\n")

captions_dict, csv_rows = {}, []
skipped = Counter()
start = time.time()

for images, labels, filenames in dataloader:
    pil_images = [transforms.ToPILImage()(img).convert("RGB") for img in images]
    try:
        captions = generate_captions(pil_images)
        for name, caption, label in zip(filenames, captions, labels):
            if is_food_caption(caption):
                final = f"{caption} (label: {label})"
                captions_dict[name] = final
                csv_rows.append([name, final])
                print(f"[{len(captions_dict)}/{max_images}] {name}: {final}")
            else:
                skipped[label] += 1
                print(f"⚠️ Skipped {name}: {caption}")
        if len(captions_dict) % 1000 == 0:
            with open("checkpoint.json", "w") as ckpt:
                json.dump(captions_dict, ckpt, indent=2)
            print("💾 Saved intermediate checkpoint.")

        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error with batch {filenames}: {e}")
        continue

# === Save Output ===
with open(output_json, "w") as f_json:
    json.dump(captions_dict, f_json, indent=4)

with open(output_csv, "w", newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["filename", "caption"])
    writer.writerows(csv_rows)

print(f"✅ Captions saved to: {output_json} and {output_csv}")
print(f"⏱️ Elapsed: {time.time() - start:.2f}s")
print("🔍 Skipped captions by class:")
for cls, count in skipped.items():
    print(f"{cls}: {count}")
