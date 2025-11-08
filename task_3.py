
import random
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

# -----------------------------
# Configuration / Hyperparams
# -----------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_BREAST_CANCER = False  # change to True to use breast cancer dataset (binary)
DATASET_NAME = "Breast Cancer" if LOAD_BREAST_CANCER else "Iris"

CONFIG = {
    "lr": 1e-2,
    "batch_size": 16,
    "epochs": 200,
    "hidden_dim": 16,
    "weight_decay": 1e-5,
    "patience": 15,  # early stopping patience
}

ACTIVATIONS = ["relu", "sigmoid", "tanh"]
OUTPUT_DIR = "results_perceptron"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Utilities: seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# -----------------------------
# Data loading & preprocessing
# -----------------------------
if LOAD_BREAST_CANCER:
    data = load_breast_cancer()
else:
    data = load_iris()

X = data["data"]
y = data["target"]
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

# Train/valid/test split 60/20/20
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
# Now split train_val into 75/25 to get 60/20/20 overall
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
)

# Fit scaler on train only
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"Dataset: {DATASET_NAME}")
print(f"Total samples: {n_samples}, features: {n_features}, classes: {n_classes}")
print(f"Train / Val / Test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
# class balance
(unique, counts) = np.unique(y, return_counts=True)
print("Class distribution (overall):", dict(zip(unique.tolist(), counts.tolist())))
(unique, counts) = np.unique(y_train, return_counts=True)
print("Class distribution (train):", dict(zip(unique.tolist(), counts.tolist())))

# Convert to torch tensors & dataloaders
def to_dataloader(X_arr, y_arr, batch_size, shuffle=False):
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = to_dataloader(X_train_s, y_train, CONFIG["batch_size"], shuffle=True)
val_loader = to_dataloader(X_val_s, y_val, CONFIG["batch_size"], shuffle=False)
test_loader = to_dataloader(X_test_s, y_test, CONFIG["batch_size"], shuffle=False)

# -----------------------------
# Model: single hidden layer, activation param
# -----------------------------
class SimplePerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation_name = activation.lower()
        # initialize weights in a consistent way
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for fc1 and fc2
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation_name == "relu":
            x = F.relu(x)
        elif self.activation_name == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation_name == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
        logits = self.fc2(x)  # no softmax here; we'll use CrossEntropyLoss
        return logits

# -----------------------------
# Training loop + utilities
# -----------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds = []
    trues = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = accuracy_score(trues, preds)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc

def train_for_activation(activation: str, return_history: bool = True) -> Dict:
    # set seed for reproducibility per run
    set_seed(SEED)

    model = SimplePerceptron(n_features, CONFIG["hidden_dim"], n_classes, activation).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        preds = []
        trues = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())

        train_loss = epoch_loss / len(train_loader.dataset)
        train_preds = np.concatenate(preds)
        train_trues = np.concatenate(trues)
        train_acc = accuracy_score(train_trues, train_preds)

        val_loss, val_acc = evaluate(model, val_loader, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # early stopping based on val_loss
        if val_loss + 1e-12 < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            # save best weights
            torch.save(best_state, os.path.join(OUTPUT_DIR, f"best_{activation}.pt"))
        else:
            patience_counter += 1

        # verbose
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{activation}] Epoch {epoch:03d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if patience_counter >= CONFIG["patience"]:
            print(f"[{activation}] Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best_val_loss: {best_val_loss:.4f}")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, DEVICE)

    # compute detailed test metrics
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds_batch = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds_batch)
            all_trues.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    precision = precision_score(all_trues, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_trues, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)

    results = {
        "activation": activation,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": classification_report(all_trues, all_preds, zero_division=0),
    }
    return results

# -----------------------------
# Run experiments for each activation
# -----------------------------
all_results = []
for act in ACTIVATIONS:
    print("\n" + "=" * 60)
    print(f"Running experiment with activation: {act}")
    res = train_for_activation(act)
    all_results.append(res)

# -----------------------------
# Summarize & Save results
# -----------------------------
import pandas as pd

summary_rows = []
for r in all_results:
    summary_rows.append(
        {
            "activation": r["activation"],
            "test_acc": r["test_acc"],
            "precision_macro": r["precision"],
            "recall_macro": r["recall"],
            "f1_macro": r["f1"],
            "best_epoch": r["best_epoch"],
        }
    )

summary_df = pd.DataFrame(summary_rows).sort_values("activation").reset_index(drop=True)
summary_csv = os.path.join(OUTPUT_DIR, "summary_results.csv")
summary_df.to_csv(summary_csv, index=False)
print("\nSummary results:")
print(summary_df)
print(f"\nSaved summary CSV to {summary_csv}")

# Print classification reports & confusion matrices
for r in all_results:
    print("\n" + "-" * 50)
    print(f"Activation: {r['activation']}")
    print("Test accuracy:", r["test_acc"])
    print("Precision (macro):", r["precision"])
    print("Recall (macro):", r["recall"])
    print("F1 (macro):", r["f1"])
    print("Confusion matrix:\n", r["confusion_matrix"])
    print("Classification report:\n", r["classification_report"])

# -----------------------------
# Visualization
# -----------------------------
# 1) Learning curves: loss and accuracy vs epochs for each activation
plt.figure(figsize=(8, 6))
for r in all_results:
    hist = r["history"]
    plt.plot(hist["train_loss"], label=f"{r['activation']} train")
    plt.plot(hist["val_loss"], "--", label=f"{r['activation']} val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss per epoch ({DATASET_NAME})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"))
print(f"Saved loss curves to {os.path.join(OUTPUT_DIR, 'loss_curves.png')}")
plt.show()

plt.figure(figsize=(8, 6))
for r in all_results:
    hist = r["history"]
    plt.plot(hist["train_acc"], label=f"{r['activation']} train")
    plt.plot(hist["val_acc"], "--", label=f"{r['activation']} val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Accuracy per epoch ({DATASET_NAME})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "acc_curves.png"))
print(f"Saved accuracy curves to {os.path.join(OUTPUT_DIR, 'acc_curves.png')}")
plt.show()

# 2) Bar chart: test accuracy per activation
plt.figure(figsize=(6, 4))
acts = [r["activation"] for r in all_results]
accs = [r["test_acc"] for r in all_results]
plt.bar(acts, accs)
plt.ylim(0, 1.0)
plt.ylabel("Test Accuracy")
plt.title(f"Test accuracy by activation ({DATASET_NAME})")
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_bar.png"))
print(f"Saved accuracy bar chart to {os.path.join(OUTPUT_DIR, 'accuracy_bar.png')}")
plt.show()

# Optional: save confusion matrices as images
for r in all_results:
    cm = r["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix ({r['activation']})")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"confusion_{r['activation']}.png")
    plt.savefig(p)
    plt.show()
    print(f"Saved confusion matrix for {r['activation']} to {p}")

print("\nAll done. Results and plots are in the folder:", OUTPUT_DIR)
