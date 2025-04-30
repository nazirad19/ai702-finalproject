import json

# Load your top-K CLIP scored file
with open("/home/Downloads/CV703_Project/caption_scores/top_3000_captions.json", "r") as f:
    data = json.load(f)

# Sort and select top-k
top_k = 10
top_items = list(data.items())[:top_k]

llama_tasks = []

for filename, entry in top_items:
    caption = entry["caption"]
    label = caption.split("(label:")[-1].replace(")", "").strip() if "(label:" in caption else None
    clean_caption = caption.split("(label:")[0].strip()

    llama_tasks.append({
        "filename": filename,
        "label": label,
        "original_caption": clean_caption
    })

# Save to file
with open("llama_refinement_input.json", "w") as out:
    json.dump(llama_tasks, out, indent=2)

print(f"✅ Prepared {len(llama_tasks)} items for LLaMA 2 refinement.")
