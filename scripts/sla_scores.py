import torch
import json
import numpy as np

# === CONFIG ===
image_embed_path = "/home/Downloads/CV703_FoodGen_Project/image_embeddings.pt"
caption_embed_path = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_embeddings.pt"
output_path = "/home/Downloads/CV703_FoodGen_Project/finetuning_experiments/experiment_5_FoodCap_6decoder/captions_finetuned_sla_scores_from_embeddings.json"

# === Load & Normalize Embeddings ===
def load_feats(path):
    data = torch.load(path)
    feats = data["features"]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return dict(zip(data["filenames"], feats))  # {filename: feat}

image_feats_dict = load_feats(image_embed_path)
caption_feats_dict = load_feats(caption_embed_path)

# === Intersect and sort keys ===
common_filenames = sorted(set(image_feats_dict.keys()) & set(caption_feats_dict.keys()))
print(f"🔁 Found {len(common_filenames)} matching filenames.")

# === Stack features in the same order
image_feats = torch.stack([image_feats_dict[f] for f in common_filenames])
caption_feats = torch.stack([caption_feats_dict[f] for f in common_filenames])

# === Compute SLA Ranks
similarity_matrix = caption_feats @ image_feats.T  # [N, N]

sla_ranks = []
per_image = {}

for i in range(len(common_filenames)):
    sims = similarity_matrix[i]
    sorted_indices = torch.argsort(sims, descending=True)
    rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-based
    sla_ranks.append(rank)

    per_image[common_filenames[i]] = {
        "sla_rank": rank
    }

# === Summary
sla_ranks_np = np.array(sla_ranks)
summary = {
    "avg_sla_rank": float(np.mean(sla_ranks_np)),
    "sla@1": int(np.sum(sla_ranks_np == 1)),
    "sla@5": int(np.sum(sla_ranks_np <= 5)),
    "sla@10": int(np.sum(sla_ranks_np <= 10)),
    "sla@20": int(np.sum(sla_ranks_np <= 20)),
    "total_images": len(sla_ranks)
}

# === Save Output
with open(output_path, "w") as f:
    json.dump({
        "summary": summary,
        "per_image": per_image
    }, f, indent=2)

print(f"✅ SLA scores saved to: {output_path}")
print(f"📉 Avg SLA Rank: {summary['avg_sla_rank']:.2f} | @1: {summary['sla@1']} | @5: {summary['sla@5']}")
