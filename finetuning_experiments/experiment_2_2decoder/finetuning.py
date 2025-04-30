import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy

# === CONFIG ===
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
ANNOTATIONS_FILE = "captions_for_finetuning.json"
IMAGE_DIR = "/home/Downloads/CV703_Project/food-101/food-101/food-101/images"
PROMPT = "Give a short, concise description of this food image in one sentence. Do not add extra details or guess unobservable context."
BATCH_SIZE = 1
EPOCHS = 7
LR = 1e-5
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# === Dataset ===
class ManualFoodDataset(Dataset):
    def __init__(self, samples, image_dir, processor, prompt):
        self.samples = samples
        self.image_dir = image_dir
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        folder = item["label"].replace(" ", "_").lower()
        image_path = os.path.join(self.image_dir, folder, item["filename"])
        image = Image.open(image_path).convert("RGB")
        caption = item["caption"]

        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )

        qformer_input_ids = inputs["qformer_input_ids"]
        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "qformer_input_ids": qformer_input_ids.squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

# === Load Model & Freeze ===
processor = InstructBlipProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)

# Freeze all, then unfreeze last 2 decoder blocks + lm_head
for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if any(f"decoder.block.{i}" in name for i in [20, 21]) or "lm_head" in name:
        param.requires_grad = True

model.to(device)

# === Load Data ===
with open(ANNOTATIONS_FILE, "r") as f:
    data = json.load(f)

samples = [{"filename": k, **v} for k, v in data.items()]
train_data, val_data = train_test_split(samples, test_size=0.1, random_state=SEED)
train_loader = DataLoader(ManualFoodDataset(train_data, IMAGE_DIR, processor, PROMPT), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ManualFoodDataset(val_data, IMAGE_DIR, processor, PROMPT), batch_size=BATCH_SIZE)

# === Optimizer ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# === Training Loop ===
best_val_loss = float("inf")
save_dir = "./instructblip_finetuned_last2_decoder"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        qformer_input_ids = batch["qformer_input_ids"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            qformer_input_ids=qformer_input_ids,
            labels=labels
        )

        loss = cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"🔥 Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} complete | Avg Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            qformer_input_ids = batch["qformer_input_ids"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                qformer_input_ids=qformer_input_ids,
                labels=labels
            )

            loss = cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🧪 Validation Loss (epoch {epoch+1}): {avg_val_loss:.4f}")

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"💾 Best model saved (val_loss = {best_val_loss:.4f}) to: {save_dir}")

# ✅ Save final model no matter what
final_save_dir = "./instructblip_finetuned_last2_decoder_final"
model.save_pretrained(final_save_dir)
processor.save_pretrained(final_save_dir)
print(f"💾 Final model also saved to: {final_save_dir}")
