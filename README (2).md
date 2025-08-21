# Selective Parameter Unlearning for Fine‑tuned Models (GPT‑2 on IMDb)

This repository contains the code and paper for a framework that **identifies and removes low-importance parameters from fine‑tuned models** while **retaining task knowledge**, demonstrated on **GPT‑2** for **IMDb sentiment analysis**. Two complementary approaches are implemented:

1. **3‑Criteria Pruning** — magnitude, movement, and gradient based pruning to flag and nullify low‑impact weights, followed by brief recovery training.
2. **Knowledge Distillation** — a compact student model trained on the teacher’s soft targets using temperature‑scaled distillation loss, combined with standard supervised loss.

> Paper: *Selective Parameter Unlearning in Fine‑tuned Models: A Framework for Efficient Knowledge Retention* (PDF in this repo).

---

## Repository Structure

```
.
├── updated-unlearning.ipynb   # 3-criteria pruning / unlearning experiments
├── final-trail.ipynb          # knowledge distillation + evaluation
└── NNDL_paper.pdf             # project paper
```

## Notebooks at a Glance

### `updated-unlearning.ipynb`
- Functions: __getitem__, __init__, __len__, create_model, evaluate_accuracy, finetune_model, gradient_based_analysis, identify_useless_parameters, load_validation_dataset, magnitude_based_pruning, measure_inference_efficiency, movement_based_analysis ...
- Classes: IMDbDataset
- Key params seen: epochs={{ 3 }}, lr={{ 5e-5 }}, batch_size={{ 10, 8 }}
- Keywords: IMDB, IMDb, gpt2, imdb
- Imports: copy, datasets, matplotlib, numpy, os, psutil, seaborn, time, torch, tqdm, transformers

### `final-trail.ipynb`
- Functions: __getitem__, __init__, __len__, __repr__, _attn, _merge_heads, _split_heads, compare_and_visualize, convert_gpt2_to_sparse, count_actual_params, create_model, evaluate_model ...
- Classes: IMDbDataset, SparseGPT2Attention, SparseGPT2Block, SparseGPT2ForSequenceClassification, SparseGPT2MLP, SparseGPT2Model, SparseLinear
- Key params seen: epochs={{ 3 }}, lr={{ 5e-5 }}, batch_size={{ 16, 8 }}
- Keywords: IMDb, gpt2, imdb
- Imports: copy, datasets, gc, math, matplotlib, numpy, psutil, seaborn, time, torch, tqdm, traceback, transformers 

## Methods

### 1) 3‑Criteria Pruning (Selective Unlearning)
- **Magnitude-based**: remove weights with small absolute values.
- **Movement-based**: remove weights that changed little from pre‑ to post‑fine‑tuning.
- **Gradient-based**: remove weights with low average gradient magnitude during fine‑tuning.
- **Recovery training**: short retraining to restore performance after pruning.

### 2) Knowledge Distillation
- **Teacher**: fine‑tuned GPT‑2 on IMDb.
- **Student**: smaller network trained on teacher **soft targets** at temperature *T*, plus standard cross‑entropy.
- **Loss**: `α * CE(student, labels) + (1-α) * KL(student_T, teacher_T)`

## Dataset

- **IMDb** (50k reviews; 25k train / 25k test; binary sentiment). 
- Tokenized as integer word indices; used for binary classification.

## Setup

Create a Python environment (e.g., conda or venv) and install dependencies. Based on notebook imports, you will likely need:

```
copy
datasets
gc
matplotlib
numpy
psutil
seaborn
torch
tqdm
traceback
transformers
```

> Tip: If you are using HuggingFace/transformers and PyTorch (common for GPT‑2), install:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers datasets accelerate scikit-learn matplotlib tqdm
> ```

## How to Run

1. **Clone** this repository (or use the Git commands below to create it from local files).
2. Open the notebooks in Jupyter or VS Code.
3. Start with **`updated-unlearning.ipynb`** to reproduce 3‑criteria pruning.
4. Proceed to **`final-trail.ipynb`** for distillation and evaluation.
5. (Optional) Adjust hyperparameters: `epochs`, `learning_rate`, `batch_size`, pruning thresholds, and distillation `temperature`.

## Expected Results (from paper)
- A large fraction of parameters can be marked redundant under the 3‑criteria test.
- Distillation yields a **smaller student** with near‑teacher performance and notably lower **time/memory**.
- Track: accuracy/F1, number of active parameters, iteration time, peak memory.

## Reproducing Figures/Tables
- The notebooks include cells to compute thresholds, plot parameter retention, and compare time/memory per iteration.
- Save figures to `reports/` (you can create this folder) for inclusion in your paper or slides.

## Citation

If you use this work, please cite the paper included in this repository.

## License

Specify the license you prefer (e.g., MIT) by creating a `LICENSE` file.

## Creating a GitHub Repository (from these files)

From a local folder containing the two notebooks and the paper:

```bash
git init
git add README.md updated-unlearning.ipynb final-trail.ipynb NNDL_paper.pdf
git commit -m "Initial commit: selective parameter unlearning + distillation (GPT-2 IMDb)"
git branch -M main
# Create a new repo on GitHub named, for example, selective-unlearning-gpt2
git remote add origin https://github.com/<your-username>/selective-unlearning-gpt2.git
git push -u origin main
```

You may also add a `.gitignore` (e.g., for Python/venv/outputs) and a `requirements.txt` with the packages above.
