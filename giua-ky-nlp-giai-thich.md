# Giải thích chi tiết: giua-ky-nlp.ipynb

## Tổng quan

Jupyter Notebook này triển khai hệ thống **Phân loại cảm xúc tiếng Việt** sử dụng mô hình transformer **PhoBERT-large**. Được thiết kế cho cuộc thi Kaggle (`midtermNLP01`), notebook này tuân thủ một pipeline ML chuyên nghiệp với cross-validation, dự đoán ensemble và nhiều kỹ thuật tối ưu hóa. Nhiệm vụ là **phân loại cảm xúc 3 lớp** (nhãn: 0, 1, 2 — có thể là tiêu cực, trung lập, tích cực).

---

## Mục lục

1. [Thiết lập môi trường](#1-thiết-lập-môi-trường)
2. [Import thư viện](#2-import-thư-viện)
3. [Hằng số cấu hình](#3-hằng-số-cấu-hình)
4. [Tiền xử lý văn bản](#4-tiền-xử-lý-văn-bản)
5. [Lớp Dataset tùy chỉnh](#5-lớp-dataset-tùy-chỉnh)
6. [Optimizer với Discriminative Learning Rates](#6-optimizer-với-discriminative-learning-rates)
7. [Hàm huấn luyện](#7-hàm-huấn-luyện)
8. [Hàm đánh giá](#8-hàm-đánh-giá)
9. [Hàm dự đoán](#9-hàm-dự-đoán)
10. [Pipeline chính](#10-pipeline-chính)
11. [Tóm tắt kết quả](#11-tóm-tắt-kết-quả)
12. [Các kỹ thuật và cải tiến chính](#12-các-kỹ-thuật-và-cải-tiến-chính)

---

## 1. Thiết lập môi trường

### Khám phá dữ liệu đầu vào

Liệt kê tất cả các file đầu vào có sẵn trong môi trường Kaggle:
- `/kaggle/input/competitions/midtermNLP01/sample_submission.csv`
- `/kaggle/input/competitions/midtermNLP01/train.csv`
- `/kaggle/input/competitions/midtermNLP01/test.csv`

### Cài đặt `underthesea`

```python
!pip install underthesea
```

Cài đặt thư viện **underthesea** (v9.2.11), một công cụ NLP tiếng Việt. Điều này **rất quan trọng** vì PhoBERT được pretrain trên văn bản tiếng Việt đã được phân tách từ (word-segmented). Nếu không phân tách từ, hiệu suất của mô hình sẽ giảm đáng kể.

---

## 2. Import thư viện

| Thư viện | Mục đích |
|---------|---------|
| `torch`, `torch.nn` | Framework deep learning — mô hình, tensor, loss |
| `transformers` | Thư viện HuggingFace để load PhoBERT tokenizer và model |
| `DataLoader`, `Dataset` | Công cụ load dữ liệu PyTorch |
| `AdamW` | Optimizer với decoupled weight decay |
| `StratifiedKFold` | K-fold cross-validation giữ nguyên phân bố nhãn |
| `f1_score` | Metric đánh giá (macro-averaged F1) |
| `tqdm` | Thanh tiến trình |

### Vietnamese Word Tokenizer

Thử import Vietnamese word tokenizer. Nếu không có, code sẽ warn nhưng vẫn chạy được.

---

## 3. Hằng số cấu hình

| Hằng số | Giá trị | Mục đích |
|----------|-------|---------|
| `MODEL_NAME` | `"vinai/phobert-large"` | Mô hình BERT tiếng Việt pretrained (biến thể large, ~1.48GB) |
| `NUM_LABELS` | `3` | Ba lớp cảm xúc (0, 1, 2) |
| `MAX_LEN` | `200` | Độ dài token tối đa (cắt padding) |
| `BATCH_SIZE` | `16` | Kích thước batch (giảm cho model large để tiết kiệm GPU) |
| `ACCUMULATION_STEPS` | `2` | Gradient accumulation → effective batch = 16 × 2 = **32** |
| `EPOCHS` | `10` | Số epoch huấn luyện tối đa mỗi fold |
| `PATIENCE` | `3` | Early stopping patience |
| `LR` | `1e-5` | Learning rate cơ bản (thấp cho model large) |
| `LABEL_SMOOTHING` | `0.1` | Label smoothing để tránh overconfidence |
| `N_FOLDS` | `5` | Số fold cross-validation |
| `DEVICE` | `cuda` | Thiết bị GPU (fallback về CPU nếu không có) |
| `SEED` | `42` | Random seed để tái tạo kết quả |

---

## 4. Tiền xử lý văn bản

```python
def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)         # Collapse multiple whitespaces
    text = re.sub(r'\.{2,}', '...', text)    # Normalize ellipsis
    text = re.sub(r'!{2,}', '!!', text)      # Normalize exclamations
    text = re.sub(r'\?{2,}', '??', text)     # Normalize question marks
    if HAS_UNDERTHESEA:
        text = vi_word_tokenize(text, format="text")  # Vietnamese word segmentation
    return text
```

### Giải thích từng bước:

1. **Convert to string và strip** — xử lý các input không phải string và loại bỏ khoảng trắng đầu/cuối.
2. **Collapse whitespace** — thay thế nhiều khoảng trắng bằng 1 khoảng trắng.
3. **Normalize ellipsis** — chuỗi 2+ dấu chấm thành `...`.
4. **Normalize exclamations** — chuỗi 2+ dấu chấm thanh thành `!!`.
5. **Normalize question marks** — chuỗi 2+ dấu hỏi thành `??`.
6. **Phân tách từ tiếng Việt** — dùng `underthesea` để tách các từ ghép tiếng Việt. Ví dụ:
   - Input: `"Hà Nội là thủ đô"`
   - Output: `"Hà_Nội là thủ_đô"` (dấu gạch dưới nối các từ)

**Tại sao phân tách từ quan trọng:** PhoBERT tokenizer được train trên văn bản đã được phân tách từ. Feed văn bản chưa phân tách sẽ khiến tokenizer tách từ sai, ảnh hưởng hiệu suất.

---

## 5. Lớp Dataset tùy chỉnh

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

PyTorch `Dataset` bao gồm dữ liệu đã được tokenize:

- **`__init__`**: Lưu encodings đã tokenize (dict với `input_ids`, `attention_mask`, etc.) và labels (tùy chọn).
- **`__len__`**: Trả về số lượng mẫu.
- **`__getitem__`**: Trả về một mẫu dưới dạng dict. Dùng `.clone().detach()` để tránh shared tensor memory.

---

## 6. Optimizer với Discriminative Learning Rates

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

### Các quyết định thiết kế quan trọng:

1. **Discriminative learning rates**: Classifier head (khởi tạo mới) nhận **LR cao gấp 10 lần** (`lr * 10`) so với pretrained backbone (`lr`). 
   - Backbone đã có kiến thức hữu ích từ pretraining — chỉ cần fine-tuning.
   - Classifier head bắt đầu từ đầu — cần học nhanh hơn.

2. **Weight decay exclusion**: Bias và LayerNorm params không áp dụng weight decay. Đây là best practice vì:
   - Các params này không được regularization.
   - Áp dụng weight decay có thể gây hại cho training stability.

---

## 7. Hàm huấn luyện

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

### Chi tiết:

1. **`model.train()`** — Chuyển sang chế độ training (enable dropout, batch norm updates).

2. **`CrossEntropyLoss(label_smoothing=0.1)`** — Label smoothing ngăn model quá tự tin. Thay vì target `[1, 0, 0]`, target trở thành `[0.9, 0.05, 0.05]`.

3. **Mixed Precision Training (AMP)** — `torch.amp.autocast('cuda')` dùng FP16 để giảm ~50% GPU memory và tăng tốc computation.

4. **Gradient Accumulation** — Loss chia cho `accumulation_steps` trước backprop. Gradients chỉ được apply mỗi N steps. Simulate batch size lớn hơn (32) mà không cần thêm GPU memory.

5. **Gradient Clipping** — `clip_grad_norm_(max_norm=1.0)` ngăn exploding gradients.

6. **GradScaler** — `torch.amp.GradScaler` xử lý loss scaling cho mixed precision training.

7. **Scheduler step** — Cosine LR scheduler được step sau mỗi optimizer update.

---

## 8. Hàm đánh giá

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

### Chức năng:

1. **`model.eval()`** — Chuyển sang chế độ evaluation (disable dropout, freeze batch norm).
2. **`torch.inference_mode()`** — Tắt hoàn toàn gradient computation.
3. **Collect predictions và ground truth** — Trích xuất predicted class và true label.
4. **Trả về macro F1 score** — Tính F1 cho mỗi class và trung bình. Quan trọng cho imbalanced datasets vì nó cho equal weight cho mỗi class.

---

## 9. Hàm dự đoán

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

### Mục đích:

Trả về **softmax probabilities** (không phải class predictions cứng) cho mỗi sample. Điều này cần thiết cho **ensemble averaging** — trung bình probabilities qua nhiều models (folds) tạo predictions mạnh mẽ và chính xác hơn.

---

## 10. Pipeline chính

### Bước 1: Thiết lập & Logging

- Đặt random seed cho reproducibility.
- Tạo thư mục `saved_models`.
- In tất cả parameters để trace.

### Bước 2: Load và Clean Data

```python
df = pd.read_csv(TRAIN_DATA_PATH)
df = df.dropna(subset=["sentence", "sentiment"])
df["sentiment"] = df["sentiment"].astype(int)
```

- Load training CSV.
- Drop rows có `sentence` hoặc `sentiment` bị thiếu.
- Đảm bảo labels là integers.

**Thống kê dataset:**
- **11,322 samples** total
- Phân bố nhãn:
  - Class 0: 5,226 samples (46.2%)
  - Class 1: 501 samples (4.4%) — **imbalanced nghiêm trọng**
  - Class 2: 5,595 samples (49.4%)

### Bước 3: Tiền xử lý văn bản

Áp dụng word segmentation và text cleaning cho tất cả training sentences.

### Bước 4: Phân tích độ dài token

Phân tích phân bố độ dài token trên 500 samples:
- Mean: 13 tokens
- Median: 11 tokens
- 95th percentile: 29 tokens
- Max: 67 tokens
- `MAX_LEN` setting: 200 (bao phủ tất cả samples)

### Bước 5: Global Pre-tokenization

Tokenize **toàn bộ dataset một lần** trước khi training. Tránh re-tokenize mỗi epoch, tiết kiệm thời gian đáng kể.

### Bước 6: K-Fold Cross-Validation Training

**Stratified K-Fold** chia data thành 5 folds trong khi giữ nguyên phân bố class trong mỗi fold. Quan trọng vì class 1 chỉ có 4.4% samples.

#### Cho mỗi fold:

1. **Split data**: `train_idx` và `val_idx` từ StratifiedKFold.
2. **Create datasets & dataloaders**: Training shuffled, validation không shuffled.
3. **Load fresh model**: Mỗi fold được model mới khởi tạo từ pretrained checkpoint.
4. **Create optimizer**: Discriminative LR optimizer.
5. **Create scheduler**: Cosine annealing với 10% warmup.
6. **Create GradScaler**: Cho mixed precision training.

#### Training Loop:
- Chạy training tối đa 10 epochs.
- Sau mỗi epoch, đánh giá trên validation fold.
- **Save best model**: Nếu `val_f1` cải thiện, lưu model vào `saved_models/fold_{fold}/`.
- **Early stopping**: Nếu validation F1 không cải thiện trong 3 epochs liên tiếp, dừng sớm.

### Bước 7: Cross-Validation Results

```
Fold 1: 0.8535
Fold 2: 0.8167
Fold 3: 0.8456
Fold 4: 0.8392
Fold 5: 0.8483
Mean F1: 0.8407 (+/- 0.0128)
Best fold: saved_models/fold_0 (F1 = 0.8535)
```

Standard deviation thấp (0.0128) cho thấy hiệu suất ổn định qua các folds.

### Bước 8: Test Prediction (5-Fold Ensemble)

1. Load và preprocess test data giống như training data.
2. Tokenize test data.
3. Tạo test DataLoader.

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

Final prediction là **argmax của trung bình probabilities** qua 5 folds. Ensemble này mạnh hơn bất kỳ model đơn lẻ nào vì:
- Mỗi fold model thấy different training data.
- Trung bình probabilities smooths out individual model errors.
- Ensemble được lợi từ diverse perspectives.

### Bước 9: Lưu Submission

```python
submission = pd.DataFrame({"id": test_df["id"], "sentiment": final_preds})
submission.to_csv("submission.csv", index=False)
```

Tạo CSV với cột `id` và `sentiment` theo format của Kaggle competition.

---

## 11. Tóm tắt kết quả

### Hiệu suất mỗi Fold

| Fold | Best F1 | Epochs | Early Stopped? |
|------|---------|--------|----------------|
| 1 | 0.8535 | 10/10 | Yes |
| 2 | 0.8167 | 7/10 | Yes |
| 3 | 0.8456 | 8/10 | Yes |
| 4 | 0.8392 | 6/10 | Yes |
| 5 | 0.8483 | 9/10 | Yes |

### Metrics tổng hợp

- **Mean Macro F1**: 0.8407
- **Standard Deviation**: 0.0128
- **Best Single Fold**: Fold 1 (F1 = 0.8535)
- **Total Training Time**: ~7 hours

---

## 12. Các kỹ thuật và cải tiến chính

Code bao gồm 11 cải tiến được đánh số so với baseline:

| # | Kỹ thuật | Mục đích |
|---|---------|---------|
| 1 | **PhoBERT-large** (vs base) | Model lớn hơn với nhiều params hơn → representations tốt hơn |
| 2 | **Vietnamese word segmentation** | Khớp với format pretraining của PhoBERT |
| 3 | **MAX_LEN = 200** (vs 128) | Bắt longer contexts mà không bị truncate |
| 4 | **Label smoothing (0.1)** | Ngăn overconfidence, cải thiện generalization |
| 5 | **Gradient accumulation (2 steps)** | Effective batch size 32 mà không cần thêm GPU memory |
| 6 | **Discriminative learning rates** | Classifier head học nhanh gấp 10 lần backbone |
| 7 | **Weight decay exclusion** | Bias và LayerNorm params được loại trừ khỏi regularization |
| 8 | **Mixed precision (AMP)** | ~2× speedup và 50% memory reduction |
| 9 | **Gradient clipping (max_norm=1.0)** | Ngăn exploding gradients |
| 10 | **Cosine scheduler with warmup** | Smooth LR schedule: warmup → cosine decay |
| 11 | **Early stopping (patience=3)** | Ngăn overfitting, tiết kiệm training time |

### Tóm tắt Architecture

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
