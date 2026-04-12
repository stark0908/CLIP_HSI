# ============================================================
# DiffFormer — Complete Self-Contained Script for Kaggle
# Dataset: Indian Pines (IP) | Input: [B, 1, 50, 9, 9]
# Classes: 16 | Metrics: OA, AA, Kappa (κ)
# Source: https://github.com/mahmad000/DiffFormer (PyTorch port)
# ============================================================

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# ─────────────────────────────────────────────────────────────
# 0. CONFIG  (edit here if needed)
# ─────────────────────────────────────────────────────────────
DATASET_ABBR = "IP"
NUM_CLASSES = 16
PCA_COMPONENTS = 50
PATCH_SIZE = 9
EMBED_DIM = 64  # d_model; must be divisible by NUM_HEADS
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = EMBED_DIM * 4  # SwiGLU hidden dim
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 100
PATIENCE = 25
BATCH_SIZE = 32

# ── Paths (same structure as your existing notebook) ─────────
PROCESSED_ROOT = "/home/23dcs505/datasets/IP"
MODEL_DIR = "/home/23dcs505/best_models"
RESULTS_DIR = "/home/23dcs505/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Alfalfa",
    "Corn notill",
    "Corn mintill",
    "Corn",
    "Grass pasture",
    "Grass trees",
    "Grass pasture mowed",
    "Hay windrowed",
    "Oats",
    "Soybean notill",
    "Soybean mintill",
    "Soybean clean",
    "Wheat",
    "Woods",
    "Buildings Grass Trees Drives",
    "Stone Steel Towers",
]

print(
    f"✅ Config ready | device={DEVICE} | embed={EMBED_DIM} "
    f"| heads={NUM_HEADS} | layers={NUM_LAYERS}"
)


# ─────────────────────────────────────────────────────────────
# 1. DATASET & DATALOADERS
#    Mirrors your existing get_dataloaders() exactly:
#    • Loads X_train.pt / y_train.pt / X_test.pt / y_test.pt
#    • WeightedRandomSampler for class-imbalance handling
#    • pin_memory, persistent_workers, prefetch_factor
# ─────────────────────────────────────────────────────────────
class HyperspectralDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor.float() if X_tensor.dtype == torch.float64 else X_tensor
        # store labels as byte to save RAM; cast to long in __getitem__
        self.y = (
            y_tensor.byte()
            if (y_tensor.dtype == torch.int64 and y_tensor.max() < 256)
            else y_tensor
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx].long() if self.y.dtype == torch.uint8 else self.y[idx]
        return x, y


def get_dataloaders(pca_components: int, dataset_abbr: str, batch_size: int = 32):
    """
    Load pre-processed HSI tensors from Kaggle dataset path.
    Returns (train_loader, test_loader).
    Tensors on disk: X shape [N, 1, pca, H, W], y shape [N] (0-indexed).
    """
    proc_dir = os.path.join(PROCESSED_ROOT, f"pca_{pca_components}", dataset_abbr)
    if not os.path.exists(proc_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {proc_dir}\n"
            f"Check PROCESSED_ROOT and that the Kaggle dataset is attached."
        )

    print(f"📂 Loading {dataset_abbr} from: {proc_dir}")
    t0 = time.time()
    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))
    print(
        f"   Loaded in {time.time() - t0:.2f}s  |  "
        f"Train: {X_tr.shape}  Test: {X_te.shape}"
    )
    print(
        f"   Label range → [{y_tr.min().item()}, {y_tr.max().item()}]  "
        f"(should be 0–{NUM_CLASSES - 1})"
    )

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    # ── WeightedRandomSampler (handles class imbalance) ───────
    y_np = y_tr.numpy()
    class_counts = np.bincount(y_np, minlength=NUM_CLASSES)
    class_w = np.zeros(NUM_CLASSES, dtype=np.float32)
    class_w[class_counts > 0] = 1.0 / class_counts[class_counts > 0]
    sample_w = torch.from_numpy(class_w[y_np]).float()
    sampler = WeightedRandomSampler(
        weights=sample_w, num_samples=len(sample_w), replacement=True
    )
    print("   ⚖️  WeightedRandomSampler initialised (balanced training)")

    # ── DataLoader config ──────────────────────────────────────
    cpu_count = os.cpu_count() or 2
    num_workers = min(4, max(2, cpu_count // 2))
    pin_memory = DEVICE.type == "cuda"
    persistent_w = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_w,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    print(
        f"   📊 Train batches: {len(train_loader)} | Test batches: {len(test_loader)}"
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# 2. DIFFFORMER ARCHITECTURE
#    Faithful PyTorch port from https://github.com/mahmad000/DiffFormer
#    ① SwiGLU feed-forward (Eq. 10)
#    ② Differential MHSA — DMHSA (Eq. 7–9, 13)
#    ③ SST encoder block (LayerNorm → DMHSA → res, LN → FFN → res)
#    ④ DiffFormer: 3D-Conv tokenisation → CLS → sinusoidal PE
#                  → L × SST → classify
# ─────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU(x, g) = (x ⊙ sigmoid(g)) + x   [Eq. 10]
    Projects to 2×hidden then splits into value and gate.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        xg = self.proj(x)
        x_val, gate = xg.chunk(2, dim=-1)
        return self.out(x_val * torch.sigmoid(gate) + x_val)


class DifferentialMHSA(nn.Module):
    """
    Differential Multi-Head Self-Attention  (Eq. 7–9, 13).

    Standard scores:   S  = Q Kᵀ / √d_h
    Differential term: ΔS = S[..., 1:] − S[..., :-1]  (pad left with 0)
    Final scores:      S' = S + λ · ΔS
    Output:            Z  = softmax(S') · V
    λ is a learnable scalar initialised to 0 (warm-start = standard MHSA).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.scale = self.d_head**-0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.lam = nn.Parameter(torch.zeros(1))  # learnable λ

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads

        def _heads(t):
            return t.view(B, N, H, self.d_head).transpose(1, 2)

        Q = _heads(self.q(x))  # [B, H, N, d_h]
        K = _heads(self.k(x))
        V = _heads(self.v(x))

        S = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        # Differential correction (column-wise finite difference, Eq. 13)
        dS = F.pad(S[..., 1:] - S[..., :-1], (1, 0))  # [B,H,N,N]
        S_prime = S + self.lam * dS

        attn = self.drop(F.softmax(S_prime, dim=-1))
        Z = torch.matmul(attn, V)  # [B,H,N,d_h]
        Z = Z.transpose(1, 2).contiguous().view(B, N, C)
        return self.out(Z)


class SSTLayer(nn.Module):
    """One DiffFormer encoder block (Spatial-Spectral Transformer layer)."""

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-3)
        self.attn = DifferentialMHSA(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-3)
        self.ffn = SwiGLU(embed_dim, ff_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DiffFormer(nn.Module):
    """
    DiffFormer adapted for input [B, 1, 50, 9, 9].

    Tokenisation:
        Conv3d kernel=(50,9,9), stride=(50,9,9) → 1 spatial-spectral token
        (mirrors the paper's original 15×12×12 → 1×1×1 token design)
    Sequence:  [CLS | patch_token]  (length = 2)
    Encoding:  sinusoidal positional encoding + L × SSTLayer
    Head:      CLS → Linear(d/2) → ReLU → Dropout → Linear(num_classes)
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 1,
        spectral_bands: int = 50,
        patch_size: int = 9,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 3D-Conv patch embedding: (B,1,50,9,9) → (B, embed_dim, 1,1,1)
        ks = (spectral_bands, patch_size, patch_size)
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, kernel_size=ks, stride=ks, padding=0
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding for seq_len = 2
        self.register_buffer("pos_enc", self._sinusoidal_pe(seq_len=2, d=embed_dim))

        # Transformer encoder
        self.layers = nn.ModuleList(
            [SSTLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.norm_out = nn.LayerNorm(embed_dim, eps=1e-3)
        self.fc_hidden = nn.Linear(embed_dim, embed_dim // 2)
        self.classifier = nn.Linear(embed_dim // 2, num_classes)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d: int) -> torch.Tensor:
        pe = torch.zeros(1, seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[: d // 2])
        return pe  # [1, seq_len, d]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x : [B, 1, 50, 9, 9]  →  logits : [B, num_classes]"""
        B = x.size(0)
        tokens = self.patch_embed(x)  # [B, embed_dim, 1, 1, 1]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, 1, embed_dim]

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # [B, 2, embed_dim]
        seq = seq + self.pos_enc  # broadcast over batch

        for layer in self.layers:
            seq = layer(seq)

        cls_out = self.norm_out(seq[:, 0])  # CLS token
        cls_out = self.drop(F.relu(self.fc_hidden(cls_out)))
        return self.classifier(cls_out)  # [B, num_classes]


# ── Shape sanity check ────────────────────────────────────────
_m = DiffFormer(
    NUM_CLASSES,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
)
_x = torch.randn(4, 1, PCA_COMPONENTS, PATCH_SIZE, PATCH_SIZE)
assert _m(_x).shape == (4, NUM_CLASSES), "Shape mismatch!"
total_params = sum(p.numel() for p in _m.parameters())
trainable_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(
    f"✅ DiffFormer OK | Total params: {total_params:,} "
    f"| Trainable: {trainable_params:,}"
)
del _m, _x


# ─────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_diffformer(train_loader, test_loader):
    model = DiffFormer(
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_path = os.path.join(MODEL_DIR, "diffformer_IP_best.pth")
    best_val_loss = float("inf")
    patience_ctr = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'=' * 65}")
    print(f"  TRAINING — DiffFormer on Indian Pines")
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"{'=' * 65}")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tr_loss += loss.item()
            tr_ok += (logits.argmax(1) == y).sum().item()
            tr_n += y.size(0)

        # ── validate ────────────────────────────────────────────
        model.eval()
        va_loss = va_ok = va_n = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(X)
                        loss = criterion(logits, y)
                else:
                    logits = model(X)
                    loss = criterion(logits, y)
                va_loss += loss.item()
                va_ok += (logits.argmax(1) == y).sum().item()
                va_n += y.size(0)

        avg_tr = tr_loss / len(train_loader)
        avg_va = va_loss / len(test_loader)
        tr_acc = 100.0 * tr_ok / tr_n
        va_acc = 100.0 * va_ok / va_n

        history["train_loss"].append(avg_tr)
        history["val_loss"].append(avg_va)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        # ── save best / early-stop ──────────────────────────────
        if avg_va < best_val_loss:
            best_val_loss = avg_va
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            star = " ★"
        else:
            patience_ctr += 1
            star = ""

        if epoch % 10 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(
                f"Ep {epoch:4d}/{EPOCHS} | "
                f"Tr {avg_tr:.4f}/{tr_acc:.1f}% | "
                f"Va {avg_va:.4f}/{va_acc:.1f}%"
                f"{star} | {elapsed:.1f}min"
            )

        if patience_ctr >= PATIENCE:
            print(
                f"\n🛑 Early stop @ epoch {epoch}  (best val loss {best_val_loss:.4f})"
            )
            break

    print(f"\n✅ Training done in {(time.time() - t0) / 60:.1f} min")
    return model, history, best_path


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION  (OA / AA / κ for IEEE TGRS)
# ─────────────────────────────────────────────────────────────
def evaluate_diffformer(model, test_loader, best_path: str):
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print("✅ Best weights loaded for evaluation.")

    model.eval()
    all_preds, all_labels = [], []
    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(X)
            else:
                logits = model(X)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # OA
    oa = accuracy_score(labels, preds) * 100.0

    # Per-class accuracy → AA
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    per_class_acc = []
    for i in range(NUM_CLASSES):
        row = cm[i].sum()
        per_class_acc.append((cm[i, i] / row * 100.0) if row > 0 else 0.0)
    aa = float(np.mean(per_class_acc))

    # Cohen's κ
    kappa = cohen_kappa_score(labels, preds)

    # ── Print (IEEE TGRS style) ───────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  DiffFormer  |  Indian Pines  |  Test-set Results")
    print(f"{'=' * 65}")
    print(f"  Overall Accuracy  (OA) : {oa:.2f}%")
    print(f"  Average Accuracy  (AA) : {aa:.2f}%")
    print(f"  Cohen's Kappa     (κ)  : {kappa:.4f}")
    print(f"{'=' * 65}")
    print("\n  Per-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        print(f"    {i + 1:2d}. {name:<35s}: {acc:6.2f}%")

    # ── Save JSON ─────────────────────────────────────────────
    results = {
        "model": "DiffFormer",
        "dataset": "Indian Pines",
        "overall_accuracy": round(oa, 4),
        "average_accuracy": round(aa, 4),
        "cohen_kappa": round(float(kappa), 6),
        "per_class_accuracy": {
            name: round(a, 4) for name, a in zip(CLASS_NAMES, per_class_acc)
        },
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(labels),
    }
    out_json = os.path.join(RESULTS_DIR, "diffformer_IP_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results  → {out_json}")

    # ── Confusion matrix plot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=[str(i + 1) for i in range(NUM_CLASSES)],
        yticklabels=[str(i + 1) for i in range(NUM_CLASSES)],
    )
    ax.set_title("DiffFormer – Indian Pines – Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "diffformer_IP_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix → {cm_path}")

    return results, cm, per_class_acc


# ─────────────────────────────────────────────────────────────
# 5. TRAINING HISTORY PLOT
# ─────────────────────────────────────────────────────────────
def plot_history(history: dict):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, history["train_loss"], label="Train")
    axes[0].plot(ep, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train")
    axes[1].plot(ep, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DiffFormer Training History – Indian Pines", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "diffformer_IP_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📈 Training history → {path}")


# ─────────────────────────────────────────────────────────────
# 6. MAIN — build loaders → train → evaluate
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STEP 1/3 — Loading data")
print("=" * 65)
train_loader, test_loader = get_dataloaders(
    PCA_COMPONENTS, DATASET_ABBR, batch_size=BATCH_SIZE
)

print("\n" + "=" * 65)
print("  STEP 2/3 — Training DiffFormer")
print("=" * 65)
diffformer_model, history, best_path = train_diffformer(train_loader, test_loader)

print("\n" + "=" * 65)
print("  STEP 3/3 — Evaluating")
print("=" * 65)
results, cm, per_class_acc = evaluate_diffformer(
    diffformer_model, test_loader, best_path
)

plot_history(history)

print("\n✅ All done! Outputs saved to /kaggle/working/results/")
