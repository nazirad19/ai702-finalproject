import torch
from torch.utils.data import Dataset, DataLoader
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from datasets import load_dataset
import random

# === CONFIG ===
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
PROMPT = "Describe only what is clearly visible in the food image in one sentence. Do not add context or assumptions."
BATCH_SIZE = 2
EPOCHS = 3
LR = 3e-5
LR_DECAY_EPOCH = 2
SEED = 42
SUBSET_SIZE = 5000  # now safe and spicy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)

# === Load Dataset (Lazy Mode) ===
print("📦 Loading Food500Cap...")
dataset = load_dataset("advancedcv/Food500Cap_train")["train"]

# === Sample indices with full label coverage ===
print("🎲 Sampling one image per class + random extras...")
indices = list(range(len(dataset)))
random.shuffle(indices)

label_seen = set()
selected_indices = []

for idx in indices:
    label = dataset[idx]["cat"]
    if label not in label_seen:
        label_seen.add(label)
        selected_indices.append(idx)
    if len(label_seen) >= 500:  # assuming 500 labels
        break

remaining = [i for i in indices if i not in selected_indices]
selected_indices += remaining[:SUBSET_SIZE - len(selected_indices)]

# Now select only those samples (this avoids loading the rest)
dataset = dataset.select(selected_indices)

print(f"✅ Subset finalized: {len(dataset)} samples | {len(label_seen)} unique labels")

# === Dataset Wrapper ===
class Food500CapDataset(Dataset):
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

        encoding = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        labels = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        ).input_ids

        encoding["labels"] = labels
        return {k: v.squeeze(0) for k, v in encoding.items()}

# === Load Model and Freeze Most of It ===
processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)

for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if any(f"decoder.block.{i}" in name for i in range(16, 22)) or "lm_head" in name:
        param.requires_grad = True

model.to(device)

# === DataLoader ===
dset = Food500CapDataset(dataset, processor, PROMPT)
train_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)

# === Optimizer ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# === Training Loop ===
print("🚀 Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0

    if epoch == LR_DECAY_EPOCH:
        for g in optimizer.param_groups:
            g["lr"] = LR * 0.1
        print(f"📉 Learning rate decayed to {LR * 0.1}")

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"][batch["labels"] == processor.tokenizer.pad_token_id] = -100

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if step % 50 == 0:
            print(f"🔥 Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} complete | Avg Loss: {total_loss / len(train_loader):.4f}")

# === Save It ===
save_path = "./instructblip_finetuned_balanced_subset"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print(f"🎉 Finetuning complete. Model saved to {save_path}")