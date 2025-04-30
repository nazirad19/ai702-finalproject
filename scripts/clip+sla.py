import json

# === Paths ===
clip_path = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_clip_scores.json"
sla_path = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_sla_scores_from_embeddings.json"
output_path = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_combined_clip_sla_scores.json"

# === Load both files ===
with open(clip_path, "r") as f:
    clip_data = json.load(f)

with open(sla_path, "r") as f:
    sla_data = json.load(f)

per_image = {}
clip_scores = []
sla_ranks = []

# === Combine per-image results ===
for filename, clip_info in clip_data.items():
    if filename in sla_data["per_image"]:
        clip_score = clip_info["clip_score"]
        sla_rank = sla_data["per_image"][filename]["sla_rank"]
        caption = clip_info["caption"]

        per_image[filename] = {
            "caption": caption,
            "clip_score": clip_score,
            "sla_rank": sla_rank
        }

        clip_scores.append(clip_score)
        sla_ranks.append(sla_rank)

# === Compute summary stats ===
import numpy as np

clip_scores_np = np.array(clip_scores)
sla_ranks_np = np.array(sla_ranks)

summary = {
    "avg_clip_score": float(np.mean(clip_scores_np)),
    "avg_sla_rank": float(np.mean(sla_ranks_np)),
    "sla@1": int(np.sum(sla_ranks_np == 1)),
    "sla@5": int(np.sum(sla_ranks_np <= 5)),
    "sla@10": int(np.sum(sla_ranks_np <= 10)),
    "sla@20": int(np.sum(sla_ranks_np <= 20)),
    "clip>0.2": int(np.sum(clip_scores_np > 0.2)),
    "clip>0.28": int(np.sum(clip_scores_np > 0.28)),
    "total_images": len(per_image)
}

# === Save the combined output ===
with open(output_path, "w") as f:
    json.dump({
        "summary": summary,
        "per_image": per_image
    }, f, indent=2)

print(f"✅ Combined results saved to {output_path}")
