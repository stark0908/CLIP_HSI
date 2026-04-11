# ============================================================
# RSSAN BASELINE — Indian Pines (PCA-50, patch 9×9, 16 classes)
# Adapted from: https://github.com/lierererniu/RSSAN-Hyperspectral-Image
# Compatible with DataLoader output: [B, 1, 50, 9, 9]
# ============================================================

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from datetime import datetime

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATASET_ABBR = "IP"
NUM_CLASSES = 16
PCA_COMPONENTS = 50  # spectral depth after PCA (== in_chanels for RSSAN)
PATCH_SIZE = 9  # spatial window (== windows for RSSAN)
OUT_CHANEL = 64  # feature maps (paper uses 64 for IP)
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 16
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
SEED = 42

CLASS_NAMES = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers",
]

processed_root = "/home/23dcs505/datasets/IP/"
output_dir = "/home/23dcs505/datasets/IP/rssan_results"
os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(
    f"Config : PCA={PCA_COMPONENTS}, patch={PATCH_SIZE}×{PATCH_SIZE}, classes={NUM_CLASSES}"
)


# ─────────────────────────────────────────────
# 1. MODEL DEFINITION  (verbatim from official repo, adapted in_chanels=50)
#    Source: https://github.com/lierererniu/RSSAN-Hyperspectral-Image/blob/main/model/network.py
# ─────────────────────────────────────────────


class Spectral_attention(nn.Module):
    """Spectral Attention Module (SeAM) — channel-wise SE-style attention."""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        y1 = self.AvgPool(X).view(X.size(0), -1)
        y2 = self.MaxPool(X).view(X.size(0), -1)
        y = self.SharedMLP(y1) + self.SharedMLP(y2)
        y = y.view(y.size(0), y.size(1), 1, 1)
        return self.sigmoid(y)


class Spatial_attention(nn.Module):
    """Spatial Attention Module (SaAM)."""

    def __init__(self, in_chanels, kernel_size, out_chanel, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chanels,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.Sigmoid()

    def forward(self, X):
        avg_out = torch.mean(X, dim=1, keepdim=True)
        max_out, _ = torch.max(X, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.act(self.conv1(y))


class RSSAN(nn.Module):
    """
    Residual Spectral-Spatial Attention Network.

    Adapted for PCA-preprocessed input:
      • in_chanels = 50  (PCA spectral depth, was 200 for raw Indian Pines)
      • windows    = 9   (spatial patch size)

    Input tensor shape expected: [B, in_chanels, H, W]
    Your DataLoader provides   : [B, 1, 50, 9, 9]
    → squeeze dim-1 before forward (handled in training loop below)
    """

    def __init__(
        self,
        feature_class,
        in_chanels,
        kernel_size,
        out_chanel,
        stride,
        padding,
        windows,
    ):
        super().__init__()
        # ── Initial spectral-spatial attention (SeAM + SaAM) ──────────────
        self.attention1 = Spectral_attention(
            in_chanels, max(1, in_chanels // 8), in_chanels
        )
        self.attention2 = Spatial_attention(2, 3, 1, 1, 1)

        # ── Residual Block 1 ──────────────────────────────────────────────
        self.conv1 = nn.Conv2d(
            in_chanels,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn3 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.attention3 = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.attention4 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu1 = nn.ReLU()

        # ── Residual Block 2 ──────────────────────────────────────────────
        self.conv4 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn4 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn5 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.attention5 = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.attention6 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu2 = nn.ReLU()

        # ── Residual Block 3 ──────────────────────────────────────────────
        self.conv6 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn6 = nn.Sequential(
            nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Conv2d(
            out_chanel,
            out_chanel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn7 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.attention7 = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.attention8 = Spatial_attention(2, 3, 1, 1, 1)
        self.relu3 = nn.ReLU()

        # ── Classifier head ───────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.full_connection = nn.Linear(out_chanel, feature_class)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, X):
        # Initial SSA with residual connection
        x1 = self.attention1(X)
        x3 = x1 * X
        x4 = self.attention2(x3) * x3 + X  # First residual connection on initial SSA

        # Low level feature extraction
        x5 = self.conv1(x4)
        x6 = self.bn1(x5)

        # Residual Block 1
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.bn3(self.conv3(x8))
        se = self.attention3(x9) * x9
        sa = self.attention4(se) * se
        x10 = self.relu1(sa + x6)  # Corrected residual connection 1

        # Residual Block 2
        x11 = self.conv4(x10)
        x12 = self.bn4(x11)
        x13 = self.bn5(self.conv5(x12))
        se1 = self.attention5(x13) * x13
        sa1 = self.attention6(se1) * se1
        x14 = self.relu2(sa1 + x10)  # Corrected residual connection 2

        # Residual Block 3
        x15 = self.conv6(x14)
        x16 = self.bn6(x15)
        x17 = self.bn7(self.conv7(x16))
        se2 = self.attention7(x17) * x17
        sa2 = self.attention8(se2) * se2
        x18 = self.relu3(sa2 + x14)  # Residual connection 3

        # Classifier head
        x_pool = self.avgpool(x18)
        x_flat = x_pool.view(x_pool.size(0), -1)
        return self.full_connection(x_flat)


# ─────────────────────────────────────────────
# 2. DATA LOADING  (reuses your existing preprocessed .pt files)
# ─────────────────────────────────────────────


class HyperspectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float() if X.dtype != torch.float32 else X
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_rssan_dataloaders(pca_components, dataset_abbr, batch_size):
    proc_dir = os.path.join(processed_root, f"pca_{pca_components}", dataset_abbr)
    assert os.path.exists(proc_dir), f"Data directory not found: {proc_dir}"

    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    print(f"Train X: {X_tr.shape}  y: {y_tr.shape}")
    print(f"Test  X: {X_te.shape}  y: {y_te.shape}")

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    # Balanced sampling for class-imbalanced IP dataset
    y_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else y_tr
    class_counts = np.bincount(y_np, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_weights = torch.tensor(class_weights[y_np], dtype=torch.float32)
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    cpu_count = os.cpu_count() or 2
    num_workers = min(4, cpu_count // 2)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────


def prepare_input(data):
    """
    DataLoader gives [B, 1, 50, 9, 9].
    RSSAN expects   [B, 50, 9, 9]  (spectral bands as 2D-conv channels).
    Squeeze the redundant dim-1.
    """
    if data.ndim == 5:
        data = data.squeeze(1)  # [B, 50, 9, 9]
    return data


def train_rssan(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_test_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "rssan_IP_best.pth")
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    print(f"\n{'=' * 60}")
    print(f"TRAINING RSSAN  —  Indian Pines")
    print(f"{'=' * 60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for data, target in train_loader:
            data = prepare_input(data).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(data)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tr_loss += loss.item()
            tr_correct += logits.argmax(1).eq(target).sum().item()
            tr_total += target.size(0)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data = prepare_input(data).to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                    logits = model(data)
                te_loss += criterion(logits, target).item()
                te_correct += logits.argmax(1).eq(target).sum().item()
                te_total += target.size(0)

        avg_tr_loss = tr_loss / len(train_loader)
        avg_te_loss = te_loss / len(test_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        te_acc = 100.0 * te_correct / te_total

        history["train_loss"].append(avg_tr_loss)
        history["test_loss"].append(avg_te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"TrLoss {avg_tr_loss:.4f} TrAcc {tr_acc:.2f}% | "
            f"TeLoss {avg_te_loss:.4f} TeAcc {te_acc:.2f}% | {elapsed:.1f}s"
        )

        # ── Early stopping + checkpoint ─────────────────────────────────
        if avg_te_loss < best_test_loss:
            best_test_loss = avg_te_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ Best model saved (loss={best_test_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nTraining complete. Best test loss: {best_test_loss:.4f}")
    return history, best_model_path


# ─────────────────────────────────────────────
# 4. EVALUATION  (OA, AA, Kappa — as required for IEEE TGRS)
# ─────────────────────────────────────────────


def evaluate_rssan(model, test_loader):
    print(f"\n{'=' * 60}")
    print("EVALUATION  —  RSSAN on Indian Pines")
    print(f"{'=' * 60}")

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data = prepare_input(data).to(device, non_blocking=True)
            logits = model(data)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(target.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── OA ──────────────────────────────────────────────────────────────
    OA = accuracy_score(all_targets, all_preds)

    # ── AA  (mean per-class recall) ──────────────────────────────────────
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == c).mean())
        else:
            per_class_acc.append(0.0)
    AA = np.mean(per_class_acc)

    # ── Kappa ────────────────────────────────────────────────────────────
    K = cohen_kappa_score(all_targets, all_preds)

    # ── Confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n{'─' * 40}")
    print(f"  Overall Accuracy  (OA) : {OA * 100:.2f}%")
    print(f"  Average Accuracy  (AA) : {AA * 100:.2f}%")
    print(f"  Kappa Coefficient  (κ) : {K:.4f}")
    print(f"{'─' * 40}")
    print(f"\nPer-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        print(f"  {i + 1:2d}. {name:<35s}: {acc * 100:.2f}%")

    # ── Save results to JSON ────────────────────────────────────────────
    results = {
        "model": "RSSAN",
        "dataset": "Indian Pines",
        "pca_components": PCA_COMPONENTS,
        "patch_size": PATCH_SIZE,
        "overall_accuracy": float(OA),
        "average_accuracy": float(AA),
        "kappa": float(K),
        "per_class_accuracy": {
            CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    results_file = os.path.join(output_dir, "rssan_IP_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_file}")

    return results, cm, per_class_acc


# ─────────────────────────────────────────────
# 5. PLOTTING UTILITIES
# ─────────────────────────────────────────────


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    axes[0].plot(epochs, history["test_loss"], "r-", lw=2, label="Test Loss")
    axes[0].set_title("Loss Curve — RSSAN (Indian Pines)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-", lw=2, label="Train Acc")
    axes[1].plot(epochs, history["test_acc"], "r-", lw=2, label="Test Acc")
    axes[1].set_title("Accuracy Curve — RSSAN (Indian Pines)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "rssan_IP_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot → {path}")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(1, NUM_CLASSES + 1),
        yticklabels=range(1, NUM_CLASSES + 1),
        cbar_kws={"label": "Samples"},
    )
    plt.title("Confusion Matrix — RSSAN (Indian Pines)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "rssan_IP_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix → {path}")


def plot_per_class_accuracy(per_class_acc):
    plt.figure(figsize=(16, 6))
    colors = [
        "steelblue" if a >= 0.90 else ("orange" if a >= 0.70 else "tomato")
        for a in per_class_acc
    ]
    bars = plt.bar(
        range(1, NUM_CLASSES + 1),
        [a * 100 for a in per_class_acc],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(
        range(1, NUM_CLASSES + 1),
        [f"{i + 1}\n{CLASS_NAMES[i][:10]}" for i in range(NUM_CLASSES)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(
        "Per-Class Accuracy — RSSAN (Indian Pines)", fontsize=14, fontweight="bold"
    )
    plt.axhline(
        y=np.mean(per_class_acc) * 100,
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"AA = {np.mean(per_class_acc) * 100:.2f}%",
    )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "rssan_IP_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy → {path}")


# ─────────────────────────────────────────────
# 6. MAIN — instantiate, train, evaluate
# ─────────────────────────────────────────────

if __name__ == "__main__" or True:  # always execute in notebook context
    # Instantiate RSSAN with PCA-50 spectral depth instead of 200
    model = RSSAN(
        feature_class=NUM_CLASSES,
        in_chanels=PCA_COMPONENTS,  # ← 50 (PCA), not 200 (raw)
        kernel_size=KERNEL_SIZE,
        out_chanel=OUT_CHANEL,
        stride=STRIDE,
        padding=PADDING,
        windows=PATCH_SIZE,  # ← 9×9 spatial patch
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RSSAN")
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Load preprocessed DataLoaders (your existing pipeline)
    train_loader, test_loader = get_rssan_dataloaders(
        PCA_COMPONENTS, DATASET_ABBR, BATCH_SIZE
    )

    # Verify tensor shape compatibility
    sample_batch, _ = next(iter(train_loader))
    print(f"\nRaw DataLoader batch shape : {sample_batch.shape}")  # [B, 1, 50, 9, 9]
    print(
        f"After prepare_input shape  : {prepare_input(sample_batch).shape}"
    )  # [B, 50, 9, 9]

    # Train
    history, best_model_path = train_rssan(model, train_loader, test_loader)

    # Load best weights before evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from: {best_model_path}")

    # Evaluate (OA / AA / Kappa)
    results, cm, per_class_acc = evaluate_rssan(model, test_loader)

    # Plots
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_per_class_accuracy(per_class_acc)

    # Final summary block for paper table
    print(f"\n{'═' * 45}")
    print(f"  RSSAN — Indian Pines  |  IEEE TGRS Table")
    print(f"{'═' * 45}")
    print(f"  OA  : {results['overall_accuracy'] * 100:.2f}%")
    print(f"  AA  : {results['average_accuracy'] * 100:.2f}%")
    print(f"  κ   : {results['kappa']:.4f}")
    print(f"{'═' * 45}")
