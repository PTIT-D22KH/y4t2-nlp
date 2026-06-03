# Detailed Explanation: giua-ky-nlp.ipynb

## Overview

This Jupyter Notebook implements a **Vietnamese Sentiment Classification** system using the **PhoBERT-large** transformer model. It is designed for a Kaggle competition (`midtermNLP01`) and follows a professional ML pipeline with cross-validation, ensemble prediction, and numerous optimization techniques. The task is a **3-class sentiment classification** (labels: 0, 1, 2 — likely negative, neutral, positive).

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Library Imports](#2-library-imports)
3. [Configuration Constants](#3-configuration-constants)
4. [Text Preprocessing](#4-text-preprocessing)
5. [Custom Dataset Class](#5-custom-dataset-class)
6. [Optimizer with Discriminative Learning Rates](#6-optimizer-with-discriminative-learning-rates)
7. [Training Function](#7-training-function)
8. [Evaluation Function](#8-evaluation-function)
9. [Prediction Function](#9-prediction-function)
10. [Main Pipeline](#10-main-pipeline)
11. [Results Summary](#11-results-summary)
12. [Key Techniques & Improvements](#12-key-techniques--improvements)

---

## 1. Environment Setup

### Input Data Discovery
```python
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

Lists all input files available in the Kaggle environment:
- `/kaggle/input/competitions/midtermNLP01/sample_submission.csv`
- `/kaggle/input/competitions/midtermNLP01/train.csv`
- `/kaggle/input/competitions/midtermNLP01/test.csv`

### Install `underthesea`
```python
!pip install underthesea
```

Installs the **underthesea** library (v9.2.11), a Vietnamese NLP toolkit. This is **critical** because PhoBERT was pretrained on word-segmented Vietnamese text. Without word segmentation, the model's performance degrades significantly.

---

## 2. Library Imports

```python
import os, re, random, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
```

| Library | Purpose |
|---------|---------|
| `torch`, `torch.nn` | Deep learning framework — model, tensors, loss |
| `transformers` | HuggingFace library for loading PhoBERT tokenizer and model |
| `DataLoader`, `Dataset` | PyTorch data loading utilities |
| `AdamW` | Optimizer with decoupled weight decay |
| `StratifiedKFold` | K-fold cross-validation preserving label distribution |
| `f1_score` | Evaluation metric (macro-averaged F1) |
| `tqdm` | Progress bars |

### Vietnamese Word Tokenizer
```python
from underthesea import word_tokenize as vi_word_tokenize
HAS_UNDERTHESEA = True
```

Attempts to import the Vietnamese word tokenizer. If unavailable, the code gracefully falls back but warns that performance will suffer.

---

## 3. Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MODEL_NAME` | `"vinai/phobert-large"` | The pre-trained Vietnamese BERT model (large variant, ~1.48GB) |
| `NUM_LABELS` | `3` | Three sentiment classes (0, 1, 2) |
| `MAX_LEN` | `200` | Maximum token sequence length (truncation/padding) |
| `BATCH_SIZE` | `16` | Mini-batch size per GPU step (reduced for large model memory) |
| `ACCUMULATION_STEPS` | `2` | Gradient accumulation steps → effective batch size = 16 × 2 = **32** |
| `EPOCHS` | `10` | Maximum training epochs per fold |
| `PATIENCE` | `3` | Early stopping patience (stop if no improvement for 3 epochs) |
| `LR` | `1e-5` | Base learning rate (lower for large model stability) |
| `LABEL_SMOOTHING` | `0.1` | Label smoothing factor to prevent overconfidence |
| `N_FOLDS` | `5` | Number of cross-validation folds |
| `DEVICE` | `cuda` | GPU device (falls back to CPU if unavailable) |
| `SEED` | `42` | Random seed for reproducibility |

---

## 4. Text Preprocessing

```python
def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)         # Collapse multiple whitespaces
    text = re.sub(r'\.{2,}', '...', text)    # Normalize ellipsis (e.g., "....." → "...")
    text = re.sub(r'!{2,}', '!!', text)      # Normalize exclamations (e.g., "!!!!" → "!!")
    text = re.sub(r'\?{2,}', '??', text)     # Normalize question marks
    if HAS_UNDERTHESEA:
        text = vi_word_tokenize(text, format="text")  # Vietnamese word segmentation
    return text
```

### What it does step by step:

1. **Convert to string and strip** — handles potential non-string inputs and removes leading/trailing whitespace.
2. **Collapse whitespace** — replaces multiple consecutive spaces/tabs/newlines with a single space.
3. **Normalize ellipsis** — any sequence of 2+ dots becomes `...` (reduces vocabulary noise).
4. **Normalize exclamations** — any sequence of 2+ `!` becomes `!!`.
5. **Normalize question marks** — any sequence of 2+ `?` becomes `??`.
6. **Vietnamese word segmentation** — uses `underthesea` to split multi-word Vietnamese compounds. For example:
   - Input: `"Hà Nội là thủ đô"`
   - Output: `"Hà_Nội là thủ_đô"` (underscores connect words that belong together)

**Why segmentation matters:** PhoBERT's tokenizer was trained on text that had already been word-segmented. Feeding unsegmented text causes the tokenizer to split words incorrectly, hurting performance.

---

## 5. Custom Dataset Class

```python
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
```

A PyTorch `Dataset` that wraps pre-tokenized data:

- **`__init__`**: Stores tokenized encodings (dict with `input_ids`, `attention_mask`, etc.) and optional labels.
- **`__len__`**: Returns the number of samples.
- **`__getitem__`**: Returns a single sample as a dict. Uses `.clone().detach()` to avoid sharing tensor memory across workers (important for `num_workers > 0` in DataLoader). Labels are converted to `torch.long` tensors for classification.

---

## 6. Optimizer with Discriminative Learning Rates

```python
def get_optimizer(model, lr=1e-5):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        # Group 1: Classifier head params (no decay)
        {"params": [...classifier...no decay...], "lr": lr * 10, "weight_decay": 0.01},
        # Group 2: Classifier head params (with decay excluded)
        {"params": [...classifier...with decay...], "lr": lr * 10, "weight_decay": 0.0},
        # Group 3: Backbone params (no decay)
        {"params": [...not classifier...no decay...], "lr": lr, "weight_decay": 0.01},
        # Group 4: Backbone params (with decay excluded)
        {"params": [...not classifier...with decay...], "lr": lr, "weight_decay": 0.0},
    ]
    return AdamW(optimizer_grouped_parameters)
```

### Key design decisions:

1. **Discriminative learning rates**: The classifier head (newly initialized, random weights) gets **10× higher learning rate** (`lr * 10`) than the pretrained backbone (`lr`). This is because:
   - The backbone already has useful knowledge from pretraining — it only needs fine-tuning.
   - The classifier head starts from scratch — it needs to learn faster.

2. **Weight decay exclusion**: Bias terms and LayerNorm parameters are excluded from weight decay. This is a standard best practice because:
   - These parameters don't benefit from regularization.
   - Applying weight decay to them can harm training stability.

3. **Four parameter groups** ensure every parameter gets the correct combination of learning rate and weight decay treatment.

---

## 7. Training Function

```python
def train_epoch(model, loader, optimizer, scheduler, scaler, accumulation_steps=1):
    model.train()
    total_loss = 0
    loss_fn = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)

    for step, batch in enumerate(pbar):
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.amp.autocast('cuda'):
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(loader)
```

### Detailed breakdown:

1. **`model.train()`** — Sets the model to training mode (enables dropout, batch norm updates).

2. **`CrossEntropyLoss(label_smoothing=0.1)`** — Uses label smoothing to prevent the model from becoming overconfident. Instead of target `[1, 0, 0]`, the target becomes `[0.9, 0.05, 0.05]`. This improves generalization.

3. **Mixed Precision Training (AMP)** — `torch.amp.autocast('cuda')` enables automatic mixed precision, using FP16 for forward/backward passes to:
   - Reduce GPU memory usage by ~50%
   - Speed up computation on modern GPUs (Tensor Cores)

4. **Gradient Accumulation** — Loss is divided by `accumulation_steps` before backpropagation. Gradients are only applied every N steps. This simulates a larger batch size (32) without requiring more GPU memory.

5. **Gradient Clipping** — `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients by scaling down gradients if their total norm exceeds 1.0.

6. **GradScaler** — `torch.amp.GradScaler` handles loss scaling for mixed precision training, preventing underflow of FP16 gradients.

7. **Scheduler step** — The cosine learning rate scheduler is stepped after each optimizer update (not each epoch).

---

## 8. Evaluation Function

```python
def eval_epoch(model, loader):
    model.eval()
    preds, gold = [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            labels = batch["labels"].to(DEVICE)
            model_batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            outputs = model(**model_batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            gold.extend(labels.cpu().numpy())

    return f1_score(gold, preds, average='macro')
```

### What it does:

1. **`model.eval()`** — Sets model to evaluation mode (disables dropout, freezes batch norm).
2. **`torch.inference_mode()`** — Disables gradient computation entirely, saving memory and speeding up inference (more efficient than `no_grad()`).
3. **Collects predictions and ground truth** — For each batch, extracts the predicted class (argmax of logits) and the true label.
4. **Returns macro F1 score** — `f1_score(..., average='macro')` computes the F1 score for each class independently and then averages them. This is important for imbalanced datasets because it gives equal weight to each class regardless of sample count.

---

## 9. Prediction Function

```python
def predict(model, loader):
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            model_batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            logits = model(**model_batch).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)
```

### Purpose:

Returns **softmax probabilities** (not hard class predictions) for each sample. This is essential for **ensemble averaging** — by averaging probabilities across multiple models (folds), the ensemble makes more calibrated and robust predictions than majority voting.

---

## 10. Main Pipeline

```python
def main():
```

### Step 1: Setup & Logging
```python
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
Sets random seed for reproducibility (affects NumPy, PyTorch, Python random). Creates the `saved_models` directory.

Prints all configuration parameters for traceability.

### Step 2: Load and Clean Data
```python
df = pd.read_csv(TRAIN_DATA_PATH)
df = df.dropna(subset=["sentence", "sentiment"])
df["sentiment"] = df["sentiment"].astype(int)
```

- Loads training CSV.
- Drops rows with missing `sentence` or `sentiment`.
- Ensures labels are integers.

**Dataset statistics** (from execution output):
- **11,322 samples** total
- Label distribution:
  - Class 0: 5,226 samples (46.2%)
  - Class 1: 501 samples (4.4%) — **severely imbalanced**
  - Class 2: 5,595 samples (49.4%)

### Step 3: Preprocess Text
```python
df["sentence"] = df["sentence"].apply(preprocess_text)
```

Applies word segmentation and text cleaning to all training sentences.

### Step 4: Token Length Analysis
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
lengths = [len(tokenizer.encode(text)) for text in df["sentence"].tolist()[:500]]
```

Analyzes token length distribution on a sample of 500 texts:
- Mean: 13 tokens
- Median: 11 tokens
- 95th percentile: 29 tokens
- Max: 67 tokens
- `MAX_LEN` setting: 200 (generous — covers all samples with room)

### Step 5: Global Pre-tokenization
```python
full_encodings = tokenizer(
    df["sentence"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)
```

Tokenizes the **entire dataset once** before training. This avoids re-tokenizing at every epoch, saving significant time. The tokenizer produces:
- `input_ids`: Token indices
- `attention_mask`: 1 for real tokens, 0 for padding
- `token_type_ids` (if applicable)

### Step 6: K-Fold Cross-Validation Training

```python
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
```

**Stratified K-Fold** splits the data into 5 folds while preserving the class distribution in each fold. This is crucial because class 1 has only 4.4% of samples — random splitting could leave some folds without any class 1 examples.

#### For each fold:

1. **Split data**: `train_idx` and `val_idx` from StratifiedKFold.
2. **Create datasets & dataloaders**:
   - Training: shuffled, `batch_size=16`, `num_workers=2`, `pin_memory=True`
   - Validation: not shuffled, same batch size
3. **Load fresh model**: `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)` — each fold gets a new model initialized from the same pretrained checkpoint.
4. **Create optimizer**: Discriminative LR optimizer.
5. **Create scheduler**: Cosine annealing with 10% warmup:
   - LR starts at 0, linearly increases to `LR` over 10% of total steps.
   - Then follows a cosine decay curve to near-zero.
6. **Create GradScaler**: For mixed precision training.

#### Training Loop (per epoch):
```python
for epoch in range(EPOCHS):
    train_loss = train_epoch(...)
    val_f1 = eval_epoch(...)
```

- Runs training for up to 10 epochs.
- After each epoch, evaluates on the validation fold.
- **Saves best model**: If `val_f1` improves, saves the model and tokenizer to `saved_models/fold_{fold}/`.
- **Early stopping**: If validation F1 doesn't improve for 3 consecutive epochs, training stops early.

### Step 7: Cross-Validation Results

After all 5 folds complete:
```
Fold 1: 0.8535
Fold 2: 0.8167
Fold 3: 0.8456
Fold 4: 0.8392
Fold 5: 0.8483
Mean F1: 0.8407 (+/- 0.0128)
Best fold: saved_models/fold_0 (F1 = 0.8407)
```

The low standard deviation (0.0128) indicates consistent performance across folds.

### Step 8: Test Prediction (5-Fold Ensemble)

```python
test_df = pd.read_csv(TEST_DATA_PATH)
test_df["sentence"] = test_df["sentence"].apply(preprocess_text)
test_encodings = tokenizer(test_df["sentence"].tolist(), ...)
test_ds = SentimentDataset(test_encodings)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
```

1. Loads and preprocesses test data identically to training data.
2. Tokenizes test data.
3. Creates test DataLoader (no labels needed).

#### Ensemble Prediction:
```python
final_probs = np.zeros((len(test_df), NUM_LABELS))
for fold in range(N_FOLDS):
    fold_model = AutoModelForSequenceClassification.from_pretrained(fold_model_path)
    final_probs += predict(fold_model, test_loader)
    del fold_model
    torch.cuda.empty_cache()
final_preds = np.argmax(final_probs, axis=1)
```

For each fold model:
1. Loads the saved model from disk.
2. Gets softmax probabilities for all test samples.
3. **Accumulates** probabilities (sums them).
4. Deletes the model and clears GPU cache (memory management).

The final prediction is the **argmax of the averaged probabilities** across all 5 folds. This ensemble approach is more robust than any single model because:
- Each fold model sees different training data.
- Averaging probabilities smooths out individual model errors.
- The ensemble benefits from diverse perspectives on the data.

### Step 9: Save Submission

```python
submission = pd.DataFrame({"id": test_df["id"], "sentiment": final_preds})
submission.to_csv("submission.csv", index=False)
```

Creates a CSV with `id` and `sentiment` columns in the format required by the Kaggle competition.

---

## 11. Results Summary

### Per-Fold Performance

| Fold | Best F1 | Epochs Trained | Early Stopped? |
|------|---------|----------------|----------------|
| 1 | 0.8535 | 10/10 | Yes (patience exhausted at epoch 10) |
| 2 | 0.8167 | 7/10 | Yes (epoch 7) |
| 3 | 0.8456 | 8/10 | Yes (epoch 8) |
| 4 | 0.8392 | 6/10 | Yes (epoch 6) |
| 5 | 0.8483 | 9/10 | Yes (epoch 9) |

### Aggregate Metrics

- **Mean Macro F1**: 0.8407
- **Standard Deviation**: 0.0128
- **Best Single Fold**: Fold 1 (F1 = 0.8535)
- **Total Training Time**: ~7 hours (02:02 to 09:12)

---

## 12. Key Techniques & Improvements

The code includes 11 numbered improvements over a baseline approach:

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | **PhoBERT-large** (vs base) | Larger model with more parameters → better representations |
| 2 | **Vietnamese word segmentation** | Matches PhoBERT's pretraining data format |
| 3 | **MAX_LEN = 200** (vs 128) | Captures longer contexts without truncation |
| 4 | **Label smoothing (0.1)** | Prevents overconfidence, improves generalization |
| 5 | **Gradient accumulation (2 steps)** | Effective batch size of 32 without extra GPU memory |
| 6 | **Discriminative learning rates** | Classifier head learns 10× faster than backbone |
| 7 | **Weight decay exclusion** | Bias and LayerNorm params excluded from regularization |
| 8 | **Mixed precision (AMP)** | ~2× speedup and 50% memory reduction |
| 9 | **Gradient clipping (max_norm=1.0)** | Prevents exploding gradients |
| 10 | **Cosine scheduler with warmup** | Smooth LR schedule: warmup → cosine decay |
| 11 | **Early stopping (patience=3)** | Prevents overfitting, saves training time |

### Architecture Summary

```
Input Text
    ↓
Text Preprocessing (clean + Vietnamese word segmentation)
    ↓
PhoBERT Tokenizer (max_length=200)
    ↓
PhoBERT-large Encoder (24 layers, ~355M params)
    ↓
Classification Head (dense → GELU → dropout → out_proj)
    ↓
3-Class Softmax Probabilities
    ↓
5-Fold Ensemble (average probabilities)
    ↓
Final Prediction (argmax)
```
