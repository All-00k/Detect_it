"""
train.py — Train BERT from scratch on a1_True.csv + a2_Fake.csv
Saves the best model weights to detect_it_weights.pt

Usage:
    cd D:\\ForMe\\Detect_IT
    python train.py

Notes:
  - Uses title + text combined (max 512 tokens) for richer context
  - BERT layers are frozen; only the classification head is trained
  - Weights are saved after every epoch (best val loss)
  - GPU will be used automatically if available (much faster)
  - CPU training: ~2-6 hours for full dataset, ~30-60 min for 10k samples
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from torch.optim import AdamW

# ── Import shared model ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import BERT_Arch

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
TRUE_CSV     = os.path.join(DATA_DIR, "a1_True.csv")
FAKE_CSV     = os.path.join(DATA_DIR, "a2_Fake.csv")
WEIGHTS_OUT  = os.path.join(os.path.dirname(__file__), "models", "detect_it_weights.pt")
MAX_LEN      = 512          # Full article context (BERT max)
BATCH_SIZE   = 8            # Keep small for CPU compatibility (16 for GPU)
EPOCHS       = 3
LR           = 2e-5
MAX_SAMPLES  = None         # Set to e.g. 10000 for faster testing, None = full dataset
RANDOM_SEED  = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Detect_IT — BERT Training")
print(f"{'='*60}")
print(f"  Device     : {device}")
print(f"  Max tokens : {MAX_LEN}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Max samples: {MAX_SAMPLES or 'full dataset'}")
print(f"{'='*60}\n")

# ── Load & prepare data ────────────────────────────────────────────────────────
print("[1/6] Loading datasets …")
if not os.path.exists(TRUE_CSV):
    raise FileNotFoundError(f"Cannot find {TRUE_CSV}\nMake sure a1_True.csv is in D:\\ForMe\\Detect_IT\\data")
if not os.path.exists(FAKE_CSV):
    raise FileNotFoundError(f"Cannot find {FAKE_CSV}\nMake sure a2_Fake.csv is in D:\\ForMe\\Detect_IT\\data")

true_df = pd.read_csv(TRUE_CSV)
fake_df = pd.read_csv(FAKE_CSV)

true_df["label"] = 0   # Real
fake_df["label"] = 1   # Fake

data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  Total articles : {len(data)}")
print(f"  Real (label=0) : {(data.label == 0).sum()}")
print(f"  Fake (label=1) : {(data.label == 1).sum()}")

# Combine title + text for richer input
data["input_text"] = data["title"].astype(str) + " [SEP] " + data["text"].astype(str)

# Optionally subsample
if MAX_SAMPLES:
    data = data.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Subsampled to  : {len(data)}")

# ── Train/val/test split ───────────────────────────────────────────────────────
print("\n[2/6] Splitting data 70/15/15 …")
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data["input_text"], data["label"],
    test_size=0.30, random_state=RANDOM_SEED, stratify=data["label"]
)
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels,
    test_size=0.50, random_state=RANDOM_SEED, stratify=temp_labels
)
print(f"  Train: {len(train_text)}  Val: {len(val_text)}  Test: {len(test_text)}")

# ── Tokenize ───────────────────────────────────────────────────────────────────
print("\n[3/6] Tokenizing with bert-base-uncased (this may take a minute) …")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_batch(texts):
    return tokenizer(
        texts.tolist(),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

tokens_train = tokenize_batch(train_text)
tokens_val   = tokenize_batch(val_text)
tokens_test  = tokenize_batch(test_text)
print("  Tokenization complete.")

# ── Build DataLoaders ──────────────────────────────────────────────────────────
print("\n[4/6] Building DataLoaders …")

def make_loader(tokens, labels, shuffle=False):
    dataset = TensorDataset(
        tokens["input_ids"],
        tokens["attention_mask"],
        torch.tensor(labels.tolist()),
    )
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

train_loader = make_loader(tokens_train, train_labels, shuffle=True)
val_loader   = make_loader(tokens_val,   val_labels)
test_loader  = make_loader(tokens_test,  test_labels)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n[5/6] Building model …")
model: nn.Module = BERT_Arch()
model.freeze_bert()    # Freeze BERT, only train the head
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params : {trainable:,} / {total:,}")

optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn     = nn.NLLLoss()

# ── Training loop ──────────────────────────────────────────────────────────────
print(f"\n[6/6] Training for {EPOCHS} epoch(s) …")
print("  (Progress is printed every 100 steps)\n")

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    print("-" * 40)

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    total_loss = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        sent_id, mask, labels = [b.to(device) for b in batch]
        model.zero_grad()
        preds = model(sent_id, mask)
        loss  = loss_fn(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
        optimizer.step()

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg     = total_loss / (step + 1)
            print(f"  Step {step+1:>5}/{len(train_loader)} | Loss: {avg:.4f} | Elapsed: {elapsed:.0f}s")

    avg_train_loss = total_loss / len(train_loader)

    # ── Validate ────────────────────────────────────────────────────────────────
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            sent_id, mask, labels = [b.to(device) for b in batch]
            preds = model(sent_id, mask)
            loss  = loss_fn(preds, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"\n  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")

    # ── Save best ───────────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), WEIGHTS_OUT)
        print(f"  ✅ Best model saved → {WEIGHTS_OUT}")
    print()

# ── Test evaluation ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Final Evaluation on Test Set")
print("=" * 60)
model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        sent_id, mask, labels = [b.to(device) for b in batch]
        preds = model(sent_id, mask)
        preds = torch.exp(preds).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
print(f"\n✅ Training complete! Weights saved to:\n   {WEIGHTS_OUT}")
print("\nNext step: run  python server.py  to start the API server.")
