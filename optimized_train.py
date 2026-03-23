import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

# --- Configuration ---
# Note: You can adjust these or keep your originals
MODEL_NAME = "vinai/phobert-base" # Assuming PhoBERT based on sample text
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 2e-5
EPOCHS = 5
NUM_LABELS = 3
OUTPUT_DIR = "./output"
TRAIN_DATA_PATH = "/home/duongvct/Documents/workspace/PTIT/Y4T2/y4t2-nlp/train.csv"
TEST_DATA_PATH = "test.csv" # Update if you have one
SEED = 42

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def train_epoch(model, loader, optimizer, scheduler, scaler):
    """Optimized with Mixed Precision (AMP)"""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        # Mixed Precision Context
        with torch.cuda.autocast():
            outputs = model(**batch)
            loss = outputs.loss
            
        # Scaling Loss
        scaler.scale(loss).backward()
        
        # Step and Unscale
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def eval_epoch(model, loader):
    """Optimized with inference_mode and faster metrics"""
    model.eval()
    preds, gold = [], []
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            labels = batch["labels"].to(DEVICE)
            # Remove labels from batch to pass to model
            model_batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            
            outputs = model(**model_batch)
            logits = outputs.logits
            
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            gold.extend(labels.cpu().numpy())
            
    # Multiclass F1 - change to 'weighted' if classes are imbalanced
    return f1_score(gold, preds, average='macro')

def predict(model, loader):
    """Prediction function that returns probabilities for ensembling"""
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            model_batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            logits = model(**model_batch).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)

def main():
    # 1. Load and Clean Data
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df.dropna(subset=["sentence", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    
    # 2. Setup Tokenizer and Fold
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # 3. GLOBAL PRE-TOKENIZATION (Optimizes speed)
    print("Tokenizing entire dataset once...")
    full_encodings = tokenizer(
        df["sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    val_scores = []
    best_overall = 0.0
    best_overall_path = None
    
    # Storage for cross-validation predictions if ensembling on test set
    test_probs_total = []

    # 4. Training Loop
    labels_array = df["sentiment"].values
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels_array)):
        print(f"\n=== Fold {fold + 1} ===")
        
        # Optimized Slice: No redundant tokenization here
        train_encodings = {k: v[train_idx] for k, v in full_encodings.items()}
        val_encodings = {k: v[val_idx] for k, v in full_encodings.items()}
        
        train_ds = SentimentDataset(train_encodings, labels_array[train_idx])
        val_ds = SentimentDataset(val_encodings, labels_array[val_idx])

        # num_workers > 0 and pin_memory=True for faster batch transfers
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        model.to(DEVICE)
        
        # Weight Decay usually helps
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        
        # Mixed Precision Scaler
        scaler = torch.cuda.GradScaler()

        best_val_f1 = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
            val_f1 = eval_epoch(model, val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} — train_loss: {train_loss:.4f} — val_f1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Save best for this fold
                fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                model.save_pretrained(fold_dir)
                tokenizer.save_pretrained(fold_dir)

        print(f"Fold {fold+1} Finished. Best val F1: {best_val_f1:.4f}")
        val_scores.append(best_val_f1)

        if best_val_f1 > best_overall:
            best_overall = best_val_f1
            best_overall_path = os.path.join(OUTPUT_DIR, f"fold_{fold}")

    print("\n=== CV Results ===")
    print(f"Mean F1: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")
    print(f"Best fold path: {best_overall_path} (F1 = {best_overall:.4f})")

    # 5. TEST PREDICTION (ENSEMBLED)
    if os.path.exists(TEST_DATA_PATH):
        print("\n=== Predicting on Test Set (5-Fold Ensemble) ===")
        test_df = pd.read_csv(TEST_DATA_PATH)
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
        
        # Predict with each fold model and average probabilities
        for fold in range(5):
            print(f"Predicting with model from fold {fold}...")
            fold_model_path = os.path.join(OUTPUT_DIR, f"fold_{fold}")
            fold_model = AutoModelForSequenceClassification.from_pretrained(fold_model_path).to(DEVICE)
            final_probs += predict(fold_model, test_loader)
            
        final_preds = np.argmax(final_probs, axis=1)
        
        submission = pd.DataFrame({"id": test_df["id"], "sentiment": final_preds})
        submission.to_csv("submission.csv", index=False)
        print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
