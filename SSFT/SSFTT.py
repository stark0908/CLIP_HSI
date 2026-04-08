# ============================================================
# SSFTT BASELINE — Indian Pines (PCA-50, patch 9×9, 16 classes)
# Adapted from: https://github.com/zgr6010/HSI_SSFTT
#
# Key shape modifications vs. original (30 bands, 13×13 patch):
#   Conv3d : kernel=(3,3,3), padding=(0,1,1)
#     => spectral: 50→48,  spatial: 9→9  (preserved via spatial padding)
#   Conv2d : kernel=(3,3),  padding=(1,1)
#     => spatial: 9→9  (preserved)
#   conv2d in_channels = 8*48 = 384  (was 8*28=224)
#   Tokenizer sequence length = 9*9 = 81  (same as original)
# ============================================================

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from datetime import datetime

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATASET_ABBR = "IP"
NUM_CLASSES = 16
PCA_COMPONENTS = 50  # spectral depth after PCA
PATCH_SIZE = 9  # spatial patch H = W

# Derived conv dimensions (verified analytically)
# Conv3d(kernel=3, pad=(0,1,1)): spec=50-3+1=48, spat=9 (preserved)
# Conv2d(kernel=3, pad=(1,1)):   spat=9 (preserved)
SPEC_OUT_3D = PCA_COMPONENTS - 3 + 1  # = 48
CONV3D_OUT_CH = 8
CONV2D_IN_CH = CONV3D_OUT_CH * SPEC_OUT_3D  # = 384
CONV2D_OUT_SPAT = PATCH_SIZE  # = 9  (padding preserves)
SEQ_LEN = CONV2D_OUT_SPAT**2  # = 81

# Transformer / tokenizer hyperparams (unchanged from paper)
NUM_TOKENS = 4
DIM = 64
DEPTH = 1
HEADS = 4
MLP_DIM = 8
DROPOUT = 0.1
EMB_DROPOUT = 0.1

# Training hyperparams
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-3
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

processed_root = "/home/23dcs505/datasets/IP"
output_dir = "/home/23dcs505/outputs"
os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(
    f"Config : PCA={PCA_COMPONENTS}, patch={PATCH_SIZE}×{PATCH_SIZE}, "
    f"conv2d_in={CONV2D_IN_CH}, seq_len={SEQ_LEN}, classes={NUM_CLASSES}"
)


# ─────────────────────────────────────────────
# 1. MODEL DEFINITION
#    Source: https://github.com/zgr6010/HSI_SSFTT/blob/main/cls_SSFTT_IP/SSFTTnet.py
#    Modified for [B, 1, 50, 9, 9] input tensors.
# ─────────────────────────────────────────────


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """Pre-norm wrapper."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float("-inf"))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(
                            LayerNormalize(
                                dim, Attention(dim, heads=heads, dropout=dropout)
                            )
                        ),
                        Residual(
                            LayerNormalize(
                                dim, MLP_Block(dim, mlp_dim, dropout=dropout)
                            )
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class SSFTTnet(nn.Module):
    """
    Spectral-Spatial Feature Tokenization Transformer (SSFTT).

    Adapted for PCA-50 + 9×9 spatial patches:
      • Conv3d: kernel=(3,3,3), padding=(0,1,1)
          spectral: 50 → 48,  spatial: 9 → 9  (spatial pad preserves size)
      • Conv2d: in_channels=8×48=384, kernel=(3,3), padding=(1,1)
          spatial: 9 → 9  (preserved)
      • Tokenizer input sequence: 9×9 = 81 spatial positions

    Input shape expected by forward(): [B, 1, 50, 9, 9]
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=NUM_CLASSES,
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        # shape params — change here if you use different PCA/patch settings
        pca_bands=PCA_COMPONENTS,  # 50
        patch_size=PATCH_SIZE,  # 9
    ):
        super().__init__()
        self.L = num_tokens
        self.cT = dim

        # ── Spectral feature extractor (3D conv) ──────────────────────────
        # kernel=(3,3,3), padding=(0,1,1):
        #   spectral dim: pca_bands - 3 + 1  (no spectral padding)
        #   spatial dims: patch_size          (padding=1 preserves)
        spec_out = pca_bands - 3 + 1  # 48 for pca=50
        conv2d_in = 8 * spec_out  # 384

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=(0, 1, 1)
            ),  # ← spatial padding added
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # ── Spatial feature extractor (2D conv) ───────────────────────────
        # kernel=(3,3), padding=(1,1): spatial size preserved → patch_size
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2d_in,
                out_channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),  # ← padding added to preserve 9×9
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # ── Gaussian-weighted tokenizer ───────────────────────────────────
        # x after conv2d: [B, 64, patch_size, patch_size]
        # after rearrange: [B, patch_size², 64]  → seq_len = patch_size²
        # token_wA / token_wV shape unchanged: [1, L, 64] and [1, 64, dim]
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)

        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

        # ── Transformer ───────────────────────────────────────────────────
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        # ── Classifier head ───────────────────────────────────────────────
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        self.apply(_weights_init)

    def forward(self, x, mask=None):
        # x: [B, 1, 50, 9, 9]
        x = self.conv3d_features(x)
        # => [B, 8, 48, 9, 9]

        x = rearrange(x, "b c h w y -> b (c h) w y")
        # => [B, 384, 9, 9]

        x = self.conv2d_features(x)
        # => [B, 64, 9, 9]

        x = rearrange(x, "b c h w -> b (h w) c")
        # => [B, 81, 64]

        # ── Softmax-weighted tokenization (Gaussian attention) ────────────
        wa = rearrange(self.token_wA, "b h w -> b w h")  # [1, 64, L]
        A = torch.einsum("bij,bjk->bik", x, wa)  # [B, 81, L]
        A = rearrange(A, "b h w -> b w h")  # [B, L, 81]
        A = A.softmax(dim=-1)  # attention over spatial tokens

        VV = torch.einsum("bij,bjk->bik", x, self.token_wV)  # [B, 81, dim]
        T = torch.einsum("bij,bjk->bik", A, VV)  # [B, L, dim]

        # ── Prepend CLS token + positional embedding ──────────────────────
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, T), dim=1)  # [B, L+1, dim]
        x += self.pos_embedding
        x = self.dropout(x)

        # ── Transformer encoder ───────────────────────────────────────────
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])  # CLS token output: [B, dim]
        return self.nn1(x)  # [B, num_classes]


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────


class HyperspectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float() if X.dtype != torch.float32 else X
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_ssftt_dataloaders(pca_components, dataset_abbr, batch_size):
    proc_dir = os.path.join(processed_root, f"pca_{pca_components}", dataset_abbr)
    assert os.path.exists(proc_dir), f"Data directory not found: {proc_dir}"

    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    print(f"Train X: {X_tr.shape}  |  y: {y_tr.shape}")
    print(f"Test  X: {X_te.shape}  |  y: {y_te.shape}")
    assert X_tr.shape[1:] == (1, 50, 9, 9), (
        f"Expected [B,1,50,9,9] patches, got {X_tr.shape}"
    )

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    # Balanced sampler for class-imbalanced IP
    y_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else y_tr
    class_counts = np.bincount(y_np, minlength=NUM_CLASSES).astype(np.float32)
    class_w = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_w = torch.tensor(class_w[y_np], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    nw = min(4, (os.cpu_count() or 2) // 2)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )

    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────


def train_ssftt(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_test_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "ssftt_IP_best.pth")
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    print(f"\n{'=' * 60}")
    print(f"TRAINING  SSFTT  —  Indian Pines")
    print(f"{'=' * 60}")
    print(f"Optimizer : Adam(lr={LR}, wd={WEIGHT_DECAY})")
    print(f"Epochs    : {EPOCHS}  |  Patience: {PATIENCE}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_correct = tr_total = 0

        for data, target in train_loader:
            # data: [B, 1, 50, 9, 9]  — passed directly, no reshape needed
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(data)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tr_loss += loss.item()
            tr_correct += logits.detach().argmax(1).eq(target).sum().item()
            tr_total += target.size(0)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        te_loss = te_correct = te_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(data)
                else:
                    logits = model(data)
                te_loss += criterion(logits, target).item()
                te_correct += logits.argmax(1).eq(target).sum().item()
                te_total += target.size(0)

        avg_tr = tr_loss / len(train_loader)
        avg_te = te_loss / len(test_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        te_acc = 100.0 * te_correct / te_total
        elapsed = time.time() - t0

        history["train_loss"].append(avg_tr)
        history["test_loss"].append(avg_te)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"TrLoss {avg_tr:.4f}  TrAcc {tr_acc:.2f}% | "
            f"TeLoss {avg_te:.4f}  TeAcc {te_acc:.2f}% | {elapsed:.1f}s"
        )

        # ── Checkpoint & early stopping ──────────────────────────────────
        if avg_te < best_test_loss:
            best_test_loss = avg_te
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ Best model saved  (loss={best_test_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}  (patience={PATIENCE})")
                break

    print(f"\nTraining complete.  Best test loss: {best_test_loss:.4f}")
    return history, best_model_path


# ─────────────────────────────────────────────
# 4. EVALUATION  (OA / AA / Kappa)
# ─────────────────────────────────────────────


def evaluate_ssftt(model, test_loader):
    print(f"\n{'=' * 60}")
    print("EVALUATION  —  SSFTT on Indian Pines")
    print(f"{'=' * 60}")

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(data)
            else:
                logits = model(data)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(target.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── OA ───────────────────────────────────────────────────────────────
    OA = accuracy_score(all_targets, all_preds)

    # ── AA  (mean per-class recall) ───────────────────────────────────────
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = all_targets == c
        per_class_acc.append((all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0)
    AA = float(np.mean(per_class_acc))

    # ── Kappa ─────────────────────────────────────────────────────────────
    K = cohen_kappa_score(all_targets, all_preds)

    cm = confusion_matrix(all_targets, all_preds)

    # ── Console output ────────────────────────────────────────────────────
    print(f"\n{'─' * 42}")
    print(f"  Overall Accuracy  (OA) : {OA * 100:.2f}%")
    print(f"  Average Accuracy  (AA) : {AA * 100:.2f}%")
    print(f"  Kappa Coefficient  (κ) : {K:.4f}")
    print(f"{'─' * 42}")
    print("\nPer-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        print(f"  {i + 1:2d}. {name:<35s}: {acc * 100:.2f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "model": "SSFTT",
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
    out_file = os.path.join(output_dir, "ssftt_IP_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_file}")

    return results, cm, per_class_acc


# ─────────────────────────────────────────────
# 5. PLOTTING UTILITIES
# ─────────────────────────────────────────────


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "b-", lw=2, label="Train")
    axes[0].plot(ep, history["test_loss"], "r-", lw=2, label="Test")
    axes[0].set_title("Loss — SSFTT (Indian Pines)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], "b-", lw=2, label="Train")
    axes[1].plot(ep, history["test_acc"], "r-", lw=2, label="Test")
    axes[1].set_title("Accuracy — SSFTT (Indian Pines)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ssftt_IP_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history → {path}")


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
    plt.title("Confusion Matrix — SSFTT (Indian Pines)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "ssftt_IP_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix → {path}")


def plot_per_class_accuracy(per_class_acc):
    colors = [
        "steelblue" if a >= 0.90 else ("orange" if a >= 0.70 else "tomato")
        for a in per_class_acc
    ]
    plt.figure(figsize=(16, 6))
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
        "Per-Class Accuracy — SSFTT (Indian Pines)", fontsize=14, fontweight="bold"
    )
    aa_val = np.mean(per_class_acc) * 100
    plt.axhline(
        y=aa_val, color="red", linestyle="--", lw=1.5, label=f"AA = {aa_val:.2f}%"
    )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ssftt_IP_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy → {path}")


# ─────────────────────────────────────────────
# 6. QUICK SHAPE SANITY CHECK (runs before training)
# ─────────────────────────────────────────────


def verify_forward_shapes():
    print("\n── Shape verification (dry run) ──")
    m = SSFTTnet().to(device)
    dummy = torch.randn(4, 1, 50, 9, 9).to(device)
    with torch.no_grad():
        out = m(dummy)
    assert out.shape == (4, NUM_CLASSES), (
        f"Expected (4, {NUM_CLASSES}), got {out.shape}"
    )
    total_p = sum(p.numel() for p in m.parameters())
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}  ✓")
    print(f"  Total params: {total_p:,}")
    del m, dummy, out
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# 7. MAIN EXECUTION
# ─────────────────────────────────────────────

if __name__ == "__main__" or True:
    # 7a. Dry-run shape check BEFORE loading data
    verify_forward_shapes()

    # 7b. Load preprocessed DataLoaders
    train_loader, test_loader = get_ssftt_dataloaders(
        PCA_COMPONENTS, DATASET_ABBR, BATCH_SIZE
    )

    # 7c. Double-check live batch shape
    sample_batch, _ = next(iter(train_loader))
    print(f"\nLive batch shape : {sample_batch.shape}")  # [B, 1, 50, 9, 9]
    assert sample_batch.shape[1:] == (1, 50, 9, 9), (
        f"Unexpected batch shape: {sample_batch.shape}"
    )
    print("Shape assertion passed ✓")

    # 7d. Instantiate final model
    model = SSFTTnet(
        in_channels=1,
        num_classes=NUM_CLASSES,
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        pca_bands=PCA_COMPONENTS,
        patch_size=PATCH_SIZE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel            : SSFTTnet")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # 7e. Train
    history, best_model_path = train_ssftt(model, train_loader, test_loader)

    # 7f. Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best weights from: {best_model_path}")

    # 7g. Evaluate — OA / AA / Kappa
    results, cm, per_class_acc = evaluate_ssftt(model, test_loader)

    # 7h. Plots
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_per_class_accuracy(per_class_acc)

    # 7i. Final IEEE TGRS table block
    print(f"\n{'═' * 45}")
    print(f"  SSFTT — Indian Pines  |  IEEE TGRS Table")
    print(f"{'═' * 45}")
    print(f"  OA  : {results['overall_accuracy'] * 100:.2f}%")
    print(f"  AA  : {results['average_accuracy'] * 100:.2f}%")
    print(f"  κ   : {results['kappa']:.4f}")
    print(f"{'═' * 45}")
