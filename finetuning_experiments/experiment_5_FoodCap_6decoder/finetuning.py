import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from datasets import load_dataset
from PIL import Image
from torch.nn.functional import cross_entropy

# === CONFIG ===
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
PROMPT = "Give a short, concise description of this food image in one sentence. Do not add extra details or guess unobservable context."
BATCH_SIZE = 1
EPOCHS = 5
LR = 1e-5
SEED = 42
SAVE_DIR = "./instructblip_finetuned_foodcap_last6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# === Load Food500Cap Dataset ===
print("📦 Loading Food500Cap dataset...")
ds = load_dataset("advancedcv/Food500Cap_train")
split = ds["train"].train_test_split(test_size=0.1, seed=SEED)
train_data = split["train"]
val_data = split["test"]

# === Dataset Class ===
class FoodCapDataset(Dataset):
    def __init__(self, data, processor, prompt):
        self.data = data
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        caption = item["caption"]

        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )

        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )["input_ids"]

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "qformer_input_ids": inputs["qformer_input_ids"].squeeze(0),
            "labels": labels.squeeze(0)
        }

# === Load Model and Freeze Most of It ===
print("🧠 Loading and freezing model...")
processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)

for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if any(f"decoder.block.{i}" in name for i in range(16, 22)) or "lm_head" in name:
        param.requires_grad = True

model.to(device)

# === Dataloaders ===
train_loader = DataLoader(FoodCapDataset(train_data, processor, PROMPT), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FoodCapDataset(val_data, processor, PROMPT), batch_size=BATCH_SIZE)

# === Optimizer ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# === Training Loop ===
print("🚀 Starting training...")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        qformer_input_ids = batch["qformer_input_ids"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

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
        if step % 20 == 0:
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
            qformer_input_ids = batch["qformer_input_ids"].to(device)
            labels = batch["labels"].to(device)
            labels[labels == processor.tokenizer.pad_token_id] = -100

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
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        print(f"💾 Best model saved (val_loss = {best_val_loss:.4f}) to: {SAVE_DIR}")

# === Final Save ===
final_save_dir = SAVE_DIR + "_final"
model.save_pretrained(final_save_dir)
processor.save_pretrained(final_save_dir)
print(f"🎉 Final model also saved to: {final_save_dir}")
