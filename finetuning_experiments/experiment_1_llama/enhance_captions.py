import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or "meta-llama/Llama-2-13b-chat-hf" if using the 13B model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Load the captions JSON
with open("top_3000_captions_llama_format.json", "r") as f:
    data = json.load(f)

enhanced_data = {}

# Define the prompt template
prompt_template = (
    "You are a language expert tasked with enhancing food image captions.\n\n"
    "Your job is to take each caption and a corresponding food label, and generate a polished, natural-sounding, vivid one-sentence description that includes the labeled food item once, smoothly and appropriately.\n\n"
    "Don't use generic openers like 'The image shows' or 'This picture depicts.' Do not repeat the food name or make assumptions not clearly stated.\n\n"
    "Avoid any mention of 'photo', 'image', 'picture', etc.\n\n"
    "Label: \"{label}\"\n"
    "Original: \"{caption}\"\n"
    "Enhanced:"
)

# Process each caption
for filename, info in tqdm(data.items(), desc="Enhancing captions"):
    caption = info["caption"]
    label = info["label"].replace("_", " ")

    prompt = prompt_template.format(label=label, caption=caption)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

    enhanced_caption = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Enhanced:")[-1].strip()
    enhanced_data[filename] = {
        "enhanced_caption": enhanced_caption,
        "label": label
    }

# Save the enhanced captions to a new JSON file
with open("top_3000_captions_enhanced.json", "w") as f:
    json.dump(enhanced_data, f, indent=2)

print("Enhanced captions saved to 'top_3000_captions_enhanced.json'")
