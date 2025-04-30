# 🍽️ FoodGen: A Vision-Language Pipeline for Food Image Captioning

This repository implements a modular pipeline for generating, refining, and evaluating fine-grained captions for food images.  
It integrates InstructBLIP, CLIP, and large language models (LLMs) to produce human-aligned captions for domain-specific settings such as food imagery.

## 🚀 Pipeline Overview

### 1. Baseline Caption Generation
- **Model**: `Salesforce/instructblip-flan-t5-xl`
- **Script**: `scripts/baseline.py`
- **Goal**: Generate concise, zero-shot captions for food images without training.

### 2. Caption Scoring with CLIP & SLA
- **Model**: `ViT-L/14` (OpenAI CLIP)
- **Scripts**:
  - `scripts/image_embeddings.py`
  - `scripts/caption_embeddings.py`
  - `scripts/clip_scores.py`
  - `scripts/sla_scores.py`
  - `scripts/clip+sla.py`
- **Goal**: Evaluate captions based on image-text embedding similarity (CLIP) and label alignment (SLA).

### 3. Optional Caption Refinement
- **Models**: `meta-llama/Llama-2-7b-chat-hf` or `GPT-4`
- **Script**: `finetuning_experiments/experiment_1_llama/enhance_captions.py`
- **Goal**: Improve pseudo-label captions while preserving factual image content.

### 4. Fine-Tuning InstructBLIP
- **Script**: `finetuning_experiments/<experiment>/finetuning.py`
- **Method**: Decoder-only fine-tuning
- **Goal**: Enhance model alignment and descriptive quality for food-specific data.

### 5. Evaluation
- **Script**: `scripts/evaluation.py`
- **Tasks**:
  - Generate test captions using fine-tuned models.
  - Evaluate with CLIP and SLA (reference-free).
  - Evaluate with BLEU, METEOR, ROUGE, and CIDEr on datasets with ground-truth.

## 📊 Metrics

| Metric                      | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| **CLIP Score**              | Cosine similarity between image and caption embeddings                  |
| **SLA Rank**                | Rank of the correct class label based on caption similarity (lower = better) |
| **SLA@1**                   | Percentage of captions where the correct label is ranked first          |
| **BLEU / ROUGE / METEOR / CIDEr** | Standard reference-based caption metrics (used with Food500Cap)     |

## 📁 Key Scripts and Files

| Script/File                                               | Description                                                   |
|------------------------------------------------------------|---------------------------------------------------------------|
| `scripts/baseline.py`                                     | Generate zero-shot captions using InstructBLIP                |
| `scripts/image_embeddings.py`                             | Generate CLIP image embeddings                                |
| `scripts/caption_embeddings.py`                           | Generate CLIP caption embeddings                              |
| `scripts/clip_scores.py`                                  | Compute CLIP scores                                           |
| `scripts/sla_scores.py`                                   | Compute SLA scores                                            |
| `scripts/clip+sla.py`                                     | Combine CLIP and SLA scores                                   |
| `scripts/finetuned_caption_generation.py`                 | Generate captions for Food101 using the fine-tuned model      |
| `scripts/generate_captions_testFood500Cap.py`             | Generate captions for the Food500Cap test set (fine-tuned)    |
| `enhance_captions.py` (under `experiment_1_llama/`)       | Refine captions using an LLM                                  |
| `scripts/evaluation.py`                                   | Evaluate generated captions using various metrics             |
| `finetuning_experiments/*/`                               | Fine-tuning configurations and results                        |


## 📂 Datasets

- **[Food101](https://www.kaggle.com/datasets/dansbecker/food-101)**  
  Used for pseudo-label generation and zero-shot captioning.

- **[Food500Cap](https://huggingface.co/datasets/advancedcv/Food500Cap/viewer/default/train)**  
  Human-annotated captions used for fine-tuning and evaluation.

- **synced_fixed_image_list.txt**  
  A text file listing selected Food101 image filenames used for consistency across runs and experiments.

## 🧠 Models Used

| Task                 | Model                                      |
|----------------------|---------------------------------------------|
| Caption Generation   | `Salesforce/instructblip-flan-t5-xl`        |
| Scoring (CLIP/SLA)   | `openai/clip-vit-large-patch14`             |
| Caption Refinement   | `meta-llama/Llama-2-7b-chat-hf` / `GPT-4`   |

## 🔗 Fine-Tuned Models

You can find the fine-tuned FoodGen models at the following links:

-  **[Experiment 1](https://huggingface.co/dnkmd/experiment1)** 
- **[Experiment 2](https://huggingface.co/dnkmd/experiment2)** 
- **[Experiment 3](https://huggingface.co/dnkmd/experiment3)**
- **[Experiment 4](https://huggingface.co/dnkmd/experiment4)**

> These include all final InstructBLIP-based models fine-tuned as part of the project.


## 🛠️ Environment Requirements
- Python 3.9+
- PyTorch with CUDA support
- Hugging Face Transformers (`transformers`) — for InstructBLIP and LLMs like LLaMA/GPT
- TorchVision (`torchvision`) — for image preprocessing and transforms
- Datasets (`datasets`) — for loading and managing Food500Cap
- OpenAI CLIP (`openai-clip`) — for image-text similarity scoring
- Pillow (`PIL`) — for image handling
- TQDM (`tqdm`) — for progress bars

## 📌 Notes
- The pipeline is modular—scripts can be reused with other datasets or models.
- Use `caption_log.txt` for tracking sample outputs during training.
- Fine-tuning experiments are organized under `finetuning_experiments/`.
- The file `synced_fixed_image_list.txt` ensures consistent sampling from Food101 across all experiments.


## 📍 Quickstart Guide

### Step 1: Generate Baseline Captions
```bash
python scripts/baseline.py
```

### Step 2: Generate Embeddings & Scores
```bash

python scripts/image_embeddings.py
python scripts/caption_embeddings.py
python scripts/clip_scores.py
python scripts/sla_scores.py
python scripts/clip+sla.py
```
### Step 3: (Optional) Refine Captions with LLM
``` bash
python finetuning_experiments/experiment_1_llama/enhance_captions.py
```
### Step 4: Fine-tune InstructBLIP
```bash
python finetuning_experiments/<experiment>/finetuning.py
```
### Step 5: Generate, Evaluate Captions (for Food500Cap), & Score Captions (Step2) 
```bash
python scripts/finetuned_caption_generation.py
python scrips/generate_captions_generation.py
python scripts/evaluation.py
```


