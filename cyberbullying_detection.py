"""
Cyberbullying Detection in Twitter Text Using Fine-tuned DistilBERT
Team: Siya Borker (23BAI1375), Aditya Kachhot (23BAI1373), Vaibhavi Jaiswal (23BAI1484)
VIT - Speech and Language Processing Project
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    MODEL_NAME    = "distilbert-base-uncased"
    MAX_LEN       = 128
    BATCH_SIZE    = 32
    EPOCHS        = 4
    LEARNING_RATE = 2e-5
    WARMUP_RATIO  = 0.1
    WEIGHT_DECAY  = 0.01
    SEED          = 42
    TEST_SIZE     = 0.15
    VAL_SIZE      = 0.15
    DATA_PATH     = "cyberbullying_tweets.csv"   # <-- update path if needed
    OUTPUT_DIR    = "model_output"
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Cyberbullying Detection -- DistilBERT Fine-tuning")
print("=" * 60)
print(f"  Device : {cfg.DEVICE}")
print(f"  Epochs : {cfg.EPOCHS}")
print(f"  Batch  : {cfg.BATCH_SIZE}")
print(f"  LR     : {cfg.LEARNING_RATE}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """Clean a single tweet string."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+",    "",    text)   # remove URLs
    text = re.sub(r"@\w+",              "",    text)   # remove @mentions
    text = re.sub(r"#(\w+)",            r"\1", text)   # keep hashtag words
    text = re.sub(r"[^a-z0-9\s!?.,']", " ",   text)   # remove special chars
    text = re.sub(r"\s+",              " ",    text)   # collapse whitespace
    return text.strip()


def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load CSV, clean text, create binary label.
    cyberbullying_type == 'not_cyberbullying'  ->  label 0
    anything else                              ->  label 1
    """
    print(f"\n[1/5] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Raw shape : {df.shape}")
    print(f"      Columns   : {df.columns.tolist()}")

    df = df.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()
    df["tweet_text"] = df["tweet_text"].apply(clean_tweet)
    df["label"]      = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
    df = df[df["tweet_text"].str.len() > 3].reset_index(drop=True)

    counts = df["label"].value_counts()
    print(f"      Clean shape      : {df.shape}")
    print(f"      Not Bullying (0) : {counts.get(0, 0)}")
    print(f"      Bullying     (1) : {counts.get(1, 0)}")
    print(f"      Imbalance ratio  : 1 : {counts.get(1,0) / max(counts.get(0,1), 1):.1f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    """Train for one epoch. Returns avg loss and macro F1."""
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        labels         = batch["labels"].to(cfg.DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss   = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if step % 50 == 0 or step == len(loader):
            print(f"    Step {step:>4}/{len(loader)} | Loss: {loss.item():.4f}", end="\r")

    print()
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def evaluate(model, loader, loss_fn):
    """Evaluate on val or test set. Returns loss, macro-f1, preds, labels."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            labels         = batch["labels"].to(cfg.DEVICE)

            logits     = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss       = loss_fn(logits, labels)
            total_loss += loss.item()

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_f1"],   "r-o", label="Val")
    axes[1].set_title("Macro F1-Score per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"      Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(labels, preds, split_name: str = "Test"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Bullying", "Bullying"],
        yticklabels=["Not Bullying", "Bullying"],
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix -- {split_name} Set", fontweight="bold")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, f"confusion_matrix_{split_name.lower()}.png")
    plt.savefig(save_path, dpi=150)
    print(f"      Saved: {save_path}")
    plt.show()


def plot_class_distribution(y_train, y_val, y_test):
    splits = ["Train", "Val", "Test"]
    not_b  = [sum(y == 0) for y in [y_train, y_val, y_test]]
    bull   = [sum(y == 1) for y in [y_train, y_val, y_test]]

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - 0.2, not_b, 0.4, label="Not Bullying", color="#4C72B0")
    bars2 = ax.bar(x + 0.2, bull,  0.4, label="Bullying",     color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title("Class Distribution per Split", fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    save_path = os.path.join(cfg.OUTPUT_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150)
    print(f"      Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation(model, loader, loss_fn, split_name: str = "Test"):
    _, _, preds, labels = evaluate(model, loader, loss_fn)

    print(f"\n{'='*60}")
    print(f"  {split_name} Set -- Full Evaluation Report")
    print(f"{'='*60}")
    print(classification_report(
        labels, preds,
        target_names=["Not Bullying", "Bullying"],
        digits=4,
    ))

    metrics = {
        "Accuracy":    accuracy_score(labels, preds),
        "Macro F1":    f1_score(labels, preds, average="macro"),
        "Weighted F1": f1_score(labels, preds, average="weighted"),
        "Binary F1":   f1_score(labels, preds, average="binary"),
        "Precision":   precision_score(labels, preds, average="macro"),
        "Recall":      recall_score(labels, preds, average="macro"),
    }
    print("  Summary:")
    for k, v in metrics.items():
        print(f"    {k:<18}: {v:.4f}")

    plot_confusion_matrix(labels, preds, split_name)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 8. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(texts: list, model, tokenizer) -> list:
    """
    Predict on raw tweet strings.
    Returns list of dicts: {text, label, confidence, prob_not_bully, prob_bully}
    """
    model.eval()
    label_map = {0: "Not Bullying", 1: "Bullying"}
    results   = []

    for text in texts:
        cleaned = clean_tweet(text)
        enc = tokenizer(
            cleaned,
            max_length=cfg.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(cfg.DEVICE),
                attention_mask=enc["attention_mask"].to(cfg.DEVICE),
            ).logits
        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        results.append({
            "text":           text,
            "label":          label_map[pred_label],
            "confidence":     f"{probs[pred_label]*100:.1f}%",
            "prob_not_bully": f"{probs[0]*100:.1f}%",
            "prob_bully":     f"{probs[1]*100:.1f}%",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Step 1: Load & Preprocess ─────────────────────────────────────────────
    df = load_and_preprocess(cfg.DATA_PATH)

    # ── Step 2: Train / Val / Test Split ──────────────────────────────────────
    print("\n[2/5] Splitting dataset...")
    X_tv, X_test, y_tv, y_test = train_test_split(
        df["tweet_text"], df["label"],
        test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED,
        stratify=df["label"],
    )
    val_ratio = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_ratio,
        random_state=cfg.SEED,
        stratify=y_tv,
    )
    print(f"      Train : {len(X_train):>6} samples")
    print(f"      Val   : {len(X_val):>6} samples")
    print(f"      Test  : {len(X_test):>6} samples")

    plot_class_distribution(y_train, y_val, y_test)

    # ── Step 3: Tokenizer & DataLoaders ───────────────────────────────────────
    print("\n[3/5] Building tokenizer and DataLoaders...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.MODEL_NAME)

    train_ds = TweetDataset(X_train, y_train, tokenizer, cfg.MAX_LEN)
    val_ds   = TweetDataset(X_val,   y_val,   tokenizer, cfg.MAX_LEN)
    test_ds  = TweetDataset(X_test,  y_test,  tokenizer, cfg.MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 4: Model + Class Weights + Optimizer ─────────────────────────────
    print("\n[4/5] Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=2
    )
    model.to(cfg.DEVICE)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(cfg.DEVICE)
    print(f"      Class weights -> Not Bullying: {class_weights[0]:.3f} | Bullying: {class_weights[1]:.3f}")

    loss_fn   = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    total_steps  = len(train_loader) * cfg.EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Step 5: Training Loop ─────────────────────────────────────────────────
    print(f"\n[5/5] Training for {cfg.EPOCHS} epochs...\n")
    history     = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch  = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"  -- Epoch {epoch}/{cfg.EPOCHS} " + "-" * 40)

        train_loss, train_f1       = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_loss, val_f1, _, _     = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"    Train -> Loss: {train_loss:.4f} | Macro F1: {train_f1:.4f}")
        print(f"    Val   -> Loss: {val_loss:.4f}   | Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(cfg.OUTPUT_DIR)
            tokenizer.save_pretrained(cfg.OUTPUT_DIR)
            print(f"    [SAVED] Best model (Val Macro F1: {best_val_f1:.4f})")

    print(f"\n  Training complete. Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history)

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print("\n  Loading best checkpoint for final evaluation...")
    best_model = DistilBertForSequenceClassification.from_pretrained(cfg.OUTPUT_DIR)
    best_model.to(cfg.DEVICE)

    test_metrics = full_evaluation(best_model, test_loader, loss_fn, split_name="Test")

    # Save metrics CSV
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "test_metrics.csv")
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved to: {metrics_path}")

    # ── Demo Inference ────────────────────────────────────────────────────────
    sample_tweets = [
        "I hope you have a wonderful day, keep smiling!",
        "you are so ugly and nobody likes you, go away forever",
        "Just finished my homework, feeling great about the results",
        "kill yourself loser, you don't deserve to live here",
        "Happy birthday! Hope your day is absolutely amazing!",
        "your religion is stupid and so are you",
        "Great match today, loved watching the game with friends",
        "go back to your country you don't belong here",
    ]

    print("\n" + "=" * 60)
    print("  Demo Inference on Sample Tweets")
    print("=" * 60)
    results = predict(sample_tweets, best_model, tokenizer)
    for r in results:
        icon = "🚨" if r["label"] == "Bullying" else "✅"
        print(f"\n  {icon} [{r['label']}] confidence: {r['confidence']}")
        print(f"     Tweet: {r['text'][:65]}")
        print(f"     Probs: Not Bullying={r['prob_not_bully']}  Bullying={r['prob_bully']}")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: ./{cfg.OUTPUT_DIR}/")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
