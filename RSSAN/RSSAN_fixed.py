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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.decomposition import PCA
import scipy.io as sio
import math
import random
import argparse
import wandb
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

processed_root = "/home/23dcs505/datasets/HSI/pca50_train5/"
output_dir = "/home/23dcs505/datasets/HSI/pca50_train5/rssan_results"
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
# 2. DATA PROCESSING UTILITIES
# ─────────────────────────────────────────────

def apply_pca(X, num_components=50):
    new_X = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_X = pca.fit_transform(new_X)
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], num_components))
    return new_X

def extract_patches(X, y, window_size=9):
    margin = int((window_size - 1) / 2)
    zero_padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    
    patches_data = []
    patches_labels = []
    
    for r in range(margin, zero_padded_X.shape[0] - margin):
        for c in range(margin, zero_padded_X.shape[1] - margin):
            if y[r - margin, c - margin] > 0:
                patch = zero_padded_X[r - margin : r + margin + 1, c - margin : c + margin + 1]
                patches_data.append(patch)
                # Labels are 1-16 in GT, convert to 0-15
                patches_labels.append(y[r - margin, c - margin] - 1)
                
    return np.array(patches_data), np.array(patches_labels)

def sample_per_class_split(X, y, n_train_per_class=10, train_ratio=None, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    train_indices = []
    test_indices = []
    
    unique_classes = np.unique(y)
    for c in unique_classes:
        indices = np.where(y == c)[0]
        np.random.shuffle(indices)
        
        if train_ratio is not None and train_ratio > 0:
            # Use percentage-based split (at least 1 sample per class)
            n_train = max(1, math.ceil(len(indices) * train_ratio))
        else:
            # Use fixed sample count
            n_train = n_train_per_class
        
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
        
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


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


def get_dataloaders(data_path, gt_path, batch_size=64, pca_components=50, patch_size=9, train_samples=10, train_ratio=None, seed=42):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # 1. Load data from .mat files
    print(f"Loading data from {data_path}...")
    data_mat = sio.loadmat(data_path)
    gt_mat = sio.loadmat(gt_path)
    
    # Handle possible different keys for IP
    hsi_cube = None
    for key in ['indian_pines_corrected', 'data', 'corrected']:
        if key in data_mat:
            hsi_cube = data_mat[key]
            break
    if hsi_cube is None:
        hsi_cube = list(data_mat.values())[-1]

    gt_map = None
    for key in ['indian_pines_gt', 'groundT', 'gt']:
        if key in gt_mat:
            gt_map = gt_mat[key]
            break
    if gt_map is None:
        gt_map = list(gt_mat.values())[-1]

    # 2. Apply PCA
    print(f"Applying PCA with {pca_components} components...")
    data_pca = apply_pca(hsi_cube, num_components=pca_components)
    
    # 3. Extract 9x9 patches
    print(f"Extracting {patch_size}x{patch_size} patches...")
    X_all, y_all = extract_patches(data_pca, gt_map, window_size=patch_size)
    
    # 4. Data split logic
    if train_ratio is not None and train_ratio > 0:
        print(f"Splitting data with {train_ratio*100:.1f}% training samples per class (Seed: {seed})...")
    else:
        print(f"Splitting data with fixed {train_samples} training samples per class (Seed: {seed})...")
    
    X_train, X_test, y_train, y_test = sample_per_class_split(X_all, y_all, 
                                                            n_train_per_class=train_samples, 
                                                            train_ratio=train_ratio, 
                                                            seed=seed)

    # 5. Print dataset statistics (while still numpy for compatibility)
    print("-" * 40)
    print(f"Dataset Summary (Seed: {seed})")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples:  {len(y_test)}")
    print("-" * 40)
    print(f"{'Class':<10} | {'Train':<8} | {'Test':<8}")
    print("-" * 40)
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    for c in unique_classes:
        tr_c = np.sum(y_train == c)
        te_c = np.sum(y_test == c)
        print(f"Class {int(c)+1:<4} | {tr_c:<8} | {te_c:<8}")
    print("-" * 40 + "\n")

    # 6. Convert to float tensors in shape [N, 1, 50, 9, 9] (compatible with prepare_input)
    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).unsqueeze(1).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).unsqueeze(1).float()
    y_test = torch.from_numpy(y_test).long()
    
    train_dataset = HyperspectralDataset(X_train, y_train)
    test_dataset = HyperspectralDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, test_loader, len(y_train), len(y_test)


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
        
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_tr_loss,
                "train/acc": tr_acc,
                "val/loss": avg_te_loss,
                "val/acc": te_acc,
                "lr": LR
            })

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSSAN Baseline for Indian Pines")
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=16)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--dataset',         type=str,   default='IP')
    parser.add_argument('--split',           type=float, default=0,
                        help="Training percentage (0-100). If 0, uses fixed 10 samples per class.")
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--name',            type=str,   default=None, help="W&B run name")
    args = parser.parse_args()

    # Apply seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize W&B
    run_name = args.name if args.name else f"RSSAN_{args.dataset}_split{args.split}_seed{args.seed}"
    wandb.init(project="RSSAN", name=run_name, config=vars(args))

    GLOBAL_SAVE_DIR = os.path.join("results", f"rssan_standalone_{args.dataset}")
    os.makedirs(GLOBAL_SAVE_DIR, exist_ok=True)

    # 1. Setup Dataloaders
    data_path = "/home/23dcs505/Prompt4HSI/Dataset/Indian_pines_corrected.mat"
    gt_path   = "/home/23dcs505/Prompt4HSI/Dataset/Indian_pines_gt.mat"
    
    train_ratio = args.split / 100.0 if args.split > 0 else None
    
    train_loader, test_loader, num_train, num_test = get_dataloaders(
        data_path, gt_path,
        batch_size=args.batch_size,
        pca_components=50,
        patch_size=9,
        train_samples=10,
        train_ratio=train_ratio,
        seed=args.seed
    )

    # 2. Instantiate Model
    model = RSSAN(
        feature_class=NUM_CLASSES,
        in_chanels=50,    # PCA components
        kernel_size=3,
        out_chanel=64,
        stride=1,
        padding=1,
        windows=9,        # Patch size
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: RSSAN | Total params: {total_params:,}")

    # 3. Train
    # Update output_dir temporarily for this training run
    output_dir = GLOBAL_SAVE_DIR 
    history, best_model_path = train_rssan(model, train_loader, test_loader)

    # 4. Evaluate
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    results, cm, per_class_acc = evaluate_rssan(model, test_loader)

    # Log Final Results and Per-Class Accuracy to W&B
    wandb.run.summary["final/OA"] = results['overall_accuracy']
    wandb.run.summary["final/AA"] = results['average_accuracy']
    wandb.run.summary["final/Kappa"] = results['kappa']
    for i, acc in enumerate(per_class_acc):
        class_name = CLASS_NAMES[i]
        wandb.run.summary[f"final/class_{i+1}_{class_name}_acc"] = acc

    # 5. Plots
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_per_class_accuracy(per_class_acc)

    print(f"\n{'═' * 45}")
    print(f"  Final Results  |  OA: {results['overall_accuracy']*100:.2f}% | AA: {results['average_accuracy']*100:.2f}%")
    print(f"{'═' * 45}")
