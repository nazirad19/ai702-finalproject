import json
from datasets import load_dataset
from nlgeval import NLGEval

# === Load ground-truth ===
print("📦 Loading ground-truth captions from HuggingFace...")
ds = load_dataset("advancedcv/Food500Cap_test")["test"]

# === Load generated captions ===
print("📦 Loading generated captions from JSON...")
with open("/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_final_food500_test.json") as f:
    generated = json.load(f)

# === Match by filename ===
print("🔍 Matching predicted and ground-truth captions...")
gen_captions = []
gt_captions = []

for fname in generated:
    try:
        index = int(fname.replace(".jpg", ""))
        gen_caption = generated[fname].split(" (label:")[0].strip()
        gt_caption = ds[index]["caption"].strip()

        gen_captions.append(gen_caption)
        gt_captions.append(gt_caption)
    except Exception as e:
        print(f"⚠️ Skipping {fname} due to error: {e}")

print(f"✅ Matched {len(gen_captions)} pairs.")

# === Run NLG-Eval ===
print("🧠 Running evaluation using NLGEval...")
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["SPICE"]
)
scores = nlgeval.compute_metrics(
    ref_list=[gt_captions],
    hyp_list=gen_captions
)

# === Print Results ===
print("\n🎯 Evaluation Results:")
for metric, score in scores.items():
    print(f"{metric.upper():<10}: {score:.4f}")