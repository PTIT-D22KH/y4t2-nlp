import os
import re
import random
import torch
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

# ============================================================
# Try to import underthesea for Vietnamese word segmentation
# PhoBERT was pretrained on word-segmented text — this is CRITICAL
# Install: pip install underthesea
# ============================================================
try:
    from underthesea import word_tokenize as vi_word_tokenize
    HAS_UNDERTHESEA = True
    print("✅ underthesea loaded — Vietnamese word segmentation enabled")
except ImportError:
    HAS_UNDERTHESEA = False
    print("⚠️  underthesea not installed. Install with: pip install underthesea")
    print("   Word segmentation is CRITICAL for PhoBERT performance!")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "vinai/phobert-large"       # [IMPROVEMENT 1] Upgraded from phobert-base
NUM_LABELS = 3
MAX_LEN = 200                            # [IMPROVEMENT 3] Increased from 128
BATCH_SIZE = 16                          # Reduced for phobert-large GPU memory
ACCUMULATION_STEPS = 2                   # [IMPROVEMENT 5] Effective batch_size = 16*2 = 32
EPOCHS = 10                              # [IMPROVEMENT 11] More epochs (early stopping will handle it)
PATIENCE = 3                             # [IMPROVEMENT 11] Early stopping patience
LR = 1e-5                               # Lower LR for large model
LABEL_SMOOTHING = 0.1                    # [IMPROVEMENT 4] Label smoothing
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DATA_PATH = '/kaggle/input/competitions/midtermNLP01/train.csv'
TEST_DATA_PATH = '/kaggle/input/competitions/midtermNLP01/test.csv'
OUTPUT_DIR = "saved_models"
SEED = 42


# ============================================================
# TEXT PREPROCESSING — [IMPROVEMENT 2]
# ============================================================
def preprocess_text(text):
    """Clean and word-segment Vietnamese text for PhoBERT"""
    text = str(text).strip()

    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)             # collapse whitespace
    text = re.sub(r'\.{2,}', '...', text)        # normalize ellipsis
    text = re.sub(r'!{2,}', '!!', text)          # normalize exclamations
    text = re.sub(r'\?{2,}', '??', text)         # normalize question marks

    # Vietnamese word segmentation (CRITICAL for PhoBERT)
    if HAS_UNDERTHESEA:
        text = vi_word_tokenize(text, format="text")

    return text


# ============================================================
# DATASET
# ============================================================
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


# ============================================================
# OPTIMIZER WITH DISCRIMINATIVE LEARNING RATES — [IMPROVEMENT 6]
# ============================================================
def get_optimizer(model, lr=1e-5):
    """
    Discriminative learning rates:
    - Classifier head: 10x base LR (fresh weights need faster learning)
    - Pretrained backbone: base LR
    - Biases & LayerNorm: no weight decay
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    optimizer_grouped_parameters = [
        # Group 1: Classifier head — higher LR
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" in n and not any(nd in n for nd in no_decay)],
            "lr": lr * 10,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" in n and any(nd in n for nd in no_decay)],
            "lr": lr * 10,
            "weight_decay": 0.0,
        },
        # Group 2: Pretrained backbone — base LR, with weight decay
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and not any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": 0.01,
        },
        # Group 3: Pretrained backbone — base LR, no weight decay
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_grouped_parameters)


# ============================================================
# TRAINING — with Label Smoothing + Gradient Accumulation
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, accumulation_steps=1):
    """Training with AMP, label smoothing, and gradient accumulation"""
    model.train()
    total_loss = 0
    loss_fn = CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # [IMPROVEMENT 4]
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)

    for step, batch in enumerate(pbar):
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.amp.autocast('cuda'):
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss = loss / accumulation_steps  # [IMPROVEMENT 5] Scale loss

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(loader)


# ============================================================
# EVALUATION
# ============================================================
def eval_epoch(model, loader):
    """Evaluation with inference_mode"""
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


# ============================================================
# PREDICTION (for ensembling)
# ============================================================
def predict(model, loader):
    """Returns softmax probabilities for ensemble averaging"""
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            model_batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            logits = model(**model_batch).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"MAX_LEN: {MAX_LEN}, BATCH_SIZE: {BATCH_SIZE}, ACCUM: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"LR: {LR}, EPOCHS: {EPOCHS}, PATIENCE: {PATIENCE}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")

    # ---- 1. Load and Clean Data ----
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df.dropna(subset=["sentence", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    print(f"\nDataset: {len(df)} samples")
    print(f"Label distribution:\n{df['sentiment'].value_counts().sort_index()}")

    # ---- 2. Preprocess Text [IMPROVEMENT 2] ----
    print("\nPreprocessing text (word segmentation)...")
    df["sentence"] = df["sentence"].apply(preprocess_text)

    # ---- 3. Check token length distribution ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    lengths = [len(tokenizer.encode(text)) for text in df["sentence"].tolist()[:500]]
    print(f"\nToken lengths (sample of 500):")
    print(f"  Mean: {np.mean(lengths):.0f}, Median: {np.median(lengths):.0f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}, Max: {max(lengths)}")
    print(f"  MAX_LEN setting: {MAX_LEN}")

    # ---- 4. Global Pre-tokenization ----
    print("\nTokenizing entire dataset once...")
    full_encodings = tokenizer(
        df["sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    labels_array = df["sentiment"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    val_scores = []
    best_overall = 0.0
    best_overall_path = None

    # ---- 5. K-Fold Training ----
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels_array)):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        # Slice pre-tokenized encodings
        train_encodings = {k: v[train_idx] for k, v in full_encodings.items()}
        val_encodings = {k: v[val_idx] for k, v in full_encodings.items()}

        train_ds = SentimentDataset(train_encodings, labels_array[train_idx])
        val_ds = SentimentDataset(val_encodings, labels_array[val_idx])

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=2
        )

        # Fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )
        model.to(DEVICE)

        # Discriminative LR optimizer [IMPROVEMENT 6]
        optimizer = get_optimizer(model, lr=LR)

        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(   # [IMPROVEMENT 10] Cosine scheduler
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler('cuda')

        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                accumulation_steps=ACCUMULATION_STEPS
            )
            val_f1 = eval_epoch(model, val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} — loss: {train_loss:.4f} — val_f1: {val_f1:.4f}", end="")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                model.save_pretrained(fold_dir)
                tokenizer.save_pretrained(fold_dir)
                print(" ✅ saved", end="")
            else:
                patience_counter += 1
                print(f" (patience {patience_counter}/{PATIENCE})", end="")

            print()

            # [IMPROVEMENT 11] Early stopping
            if patience_counter >= PATIENCE:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break

        print(f"Fold {fold+1} Best F1: {best_val_f1:.4f}")
        val_scores.append(best_val_f1)

        if best_val_f1 > best_overall:
            best_overall = best_val_f1
            best_overall_path = os.path.join(OUTPUT_DIR, f"fold_{fold}")

    # ---- 6. CV Results ----
    print(f"\n{'='*50}")
    print(f"  CV RESULTS")
    print(f"{'='*50}")
    for i, score in enumerate(val_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"  Mean F1: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")
    print(f"  Best fold: {best_overall_path} (F1 = {best_overall:.4f})")

    # ---- 7. Test Prediction (5-Fold Ensemble) ----
    if os.path.exists(TEST_DATA_PATH):
        print(f"\n{'='*50}")
        print(f"  TEST PREDICTION (5-Fold Ensemble)")
        print(f"{'='*50}")

        test_df = pd.read_csv(TEST_DATA_PATH)

        # Apply same preprocessing to test data
        test_df["sentence"] = test_df["sentence"].apply(preprocess_text)

        test_encodings = tokenizer(
            test_df["sentence"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        test_ds = SentimentDataset(test_encodings)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        final_probs = np.zeros((len(test_df), NUM_LABELS))

        for fold in range(N_FOLDS):
            fold_model_path = os.path.join(OUTPUT_DIR, f"fold_{fold}")
            if not os.path.exists(fold_model_path):
                print(f"⚠️  Fold {fold} model not found, skipping")
                continue
            print(f"Predicting with fold {fold} model...")
            fold_model = AutoModelForSequenceClassification.from_pretrained(
                fold_model_path
            ).to(DEVICE)
            final_probs += predict(fold_model, test_loader)
            del fold_model
            torch.cuda.empty_cache()

        final_preds = np.argmax(final_probs, axis=1)

        submission = pd.DataFrame({"id": test_df["id"], "sentiment": final_preds})
        submission.to_csv("submission.csv", index=False)
        print(f"\n✅ Submission saved to submission.csv ({len(submission)} rows)")
        print(f"Prediction distribution:\n{pd.Series(final_preds).value_counts().sort_index()}")


if __name__ == "__main__":
    main()
