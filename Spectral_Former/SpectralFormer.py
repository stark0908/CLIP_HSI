# ╔══════════════════════════════════════════════════════════════════╗
# ║  SpectralFormer (Patch-Wise, CAF) — Indian Pines                ║
# ║  Baseline for IEEE TGRS submission comparison                   ║
# ║  Reference: Hong et al., IEEE TGRS 2022                         ║
# ║  Official repo: github.com/danfenghong/IEEE_TGRS_SpectralFormer ║
# ╚══════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ═══════════════════════════════════════════════════════════════════════════
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score,
)
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device   : {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Configuration
# All paths, dataset metadata, and SpectralFormer hyperparameters live here.
# ═══════════════════════════════════════════════════════════════════════════

# ── Kaggle I/O paths (match your existing notebook) ──────────────────────
PROCESSED_ROOT = "/home/23dcs505/datasets/IP"
MODEL_DIR = "/home/23dcs505/best_models"
RESULTS_DIR = "/home/23dcs505/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Indian Pines metadata ─────────────────────────────────────────────────
DATASET_ABBR = "IP"
NUM_CLASSES = 16
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

# ── Pre-processed tensor dimensions ──────────────────────────────────────
# Your DataLoader yields [B, 1, 50, 9, 9].
NUM_PCA = 50  # spectral dimension after PCA
PATCH_SIZE = 9  # spatial H = W

# ── SpectralFormer hyperparameters ────────────────────────────────────────
# Source: official repo README, Indian Pines patch-wise config:
#   --patches=7 --band_patches=3 --mode='CAF' --weight_decay=5e-3
# We use patch_size=9 (your data) instead of 7.
BAND_PATCHES = 3  # spectral grouping factor  → 17 tokens from 50 (padded to 51)
EMBED_DIM = 64  # transformer hidden dim
DEPTH = 5  # transformer encoder depth
HEADS = 4  # attention heads
DIM_HEAD = 16  # dim per head
MLP_DIM = 8  # FFN hidden dim (compact by design in SpectralFormer)
DROPOUT = 0.1
EMB_DROPOUT = 0.1

# ── Training hyperparameters ──────────────────────────────────────────────
# Source: official demo.py defaults for Indian Pines
EPOCHS = 100
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 5e-3
PATIENCE = 30  # early-stopping patience
GRAD_CLIP = 1.0
MIXED_PREC = True and torch.cuda.is_available()
CACHE_FREQ = 10  # clear GPU cache every N epochs

print("Configuration loaded ✓")
print(f"  Dataset : {DATASET_ABBR}  |  Classes : {NUM_CLASSES}")
print(f"  Input   : [B, 1, {NUM_PCA}, {PATCH_SIZE}, {PATCH_SIZE}]")
print(
    f"  Tokens  : {((NUM_PCA + BAND_PATCHES - 1) // BAND_PATCHES)} spectral tokens "
    f"(band_patches={BAND_PATCHES}, padded to "
    f"{((NUM_PCA + BAND_PATCHES - 1) // BAND_PATCHES) * BAND_PATCHES})"
)
print(
    f"  Token dim: {BAND_PATCHES * PATCH_SIZE * PATCH_SIZE}  →  embed_dim={EMBED_DIM}"
)

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Dataset & DataLoaders
# Mirrors your existing notebook's get_dataloaders() exactly,
# but is fully self-contained (no external logger dependencies).
# ═══════════════════════════════════════════════════════════════════════════


class HyperspectralDataset(Dataset):
    """
    Thin wrapper over pre-loaded PyTorch tensors.
    X : [N, 1, num_pca, H, W]  float32
    y : [N]                     int64 (0-indexed)
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    dataset_abbr: str = "IP",
    pca_components: int = 50,
    batch_size: int = 64,
):
    """
    Load pre-processed tensors from disk and return
    (train_loader, test_loader).
    """
    proc_dir = os.path.join(PROCESSED_ROOT, f"pca_{pca_components}", dataset_abbr)
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"Data directory not found: {proc_dir}")

    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    # Normalise label dtype
    if y_tr.dtype == torch.uint8:
        y_tr = y_tr.long()
    if y_te.dtype == torch.uint8:
        y_te = y_te.long()

    print(f"  Train  X: {tuple(X_tr.shape)}  y: {tuple(y_tr.shape)}")
    print(f"  Test   X: {tuple(X_te.shape)}  y: {tuple(y_te.shape)}")

    # ── WeightedRandomSampler: addresses class imbalance in Indian Pines ──
    y_np = y_tr.numpy()
    counts = np.bincount(y_np, minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    s_w = torch.from_numpy(weights[y_np])
    sampler = WeightedRandomSampler(s_w, len(s_w), replacement=True)
    print("  ⚖️  WeightedRandomSampler initialised for class-balanced training")

    n_cpu = os.cpu_count() or 1
    n_work = min(4, n_cpu // 2) if n_cpu > 2 else 2
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        HyperspectralDataset(X_tr, y_tr),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=n_work,
        pin_memory=pin_mem,
        persistent_workers=(n_work > 0),
        prefetch_factor=2 if n_work > 0 else None,
        drop_last=True,
    )
    test_loader = DataLoader(
        HyperspectralDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_work,
        pin_memory=pin_mem,
        persistent_workers=(n_work > 0),
        prefetch_factor=2 if n_work > 0 else None,
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Test  batches : {len(test_loader)}")
    return train_loader, test_loader

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — SpectralFormer Model (Patch-Wise, CAF Mode)
#
# Faithful PyTorch re-implementation extracted from the official repository:
#   github.com/danfenghong/IEEE_TGRS_SpectralFormer/blob/main/vit_pytorch.py
#
# Input pipeline for your pre-processed data [B, 1, 50, 9, 9]:
#   Step 1  squeeze channel  →  [B, 50, 9, 9]
#   Step 2  flatten spatial  →  [B, 50, 81]
#   Step 3  zero-pad bands   →  [B, 51, 81]   (51 = ⌈50/3⌉×3)
#   Step 4  group 3 bands    →  [B, 17, 243]  (17 spectral tokens)
#   Step 5  linear project   →  [B, 17, 64]
#   Step 6  prepend [CLS]    →  [B, 18, 64]
#   Step 7  add pos_embed    →  [B, 18, 64]
#   Step 8  CAF Transformer  →  [B, 18, 64]
#   Step 9  [CLS] → head     →  [B, 16]
# ═══════════════════════════════════════════════════════════════════════════

# ── Building blocks ───────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation used inside each CAF block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 16, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dh = dim_head
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h, d = self.heads, self.dh
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 × [B,N,h*d]
        q, k, v = (
            t.reshape(B, N, h, d).transpose(1, 2)  # [B,h,N,d]
            for t in qkv
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,N,N]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class CAFBlock(nn.Module):
    """
    SpectralFormer Transformer Block with Cross-layer Adaptive Fusion (CAF).

    The standard ViT applies a fixed residual connection: x ← x + f(x).
    CAF replaces this with a learned per-token gate that adaptively weighs
    the transformed output h against the block's own input x_in:

        h     =  FFN( LN( Attn( LN(x) ) + x ) )  +  Attn( LN(x) ) + x
        gate  =  σ( W · cat(h, x_in) )   ∈ (0,1)^D
        x_out =  gate ⊙ h  +  (1 − gate) ⊙ x_in

    This is the core spectral-domain innovation of SpectralFormer over ViT.
    Source: Hong et al. (2022), Eq. (9-11), IEEE TGRS.
    """

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        # CAF gate: conditioned on both the block output and its input
        self.caf_gate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x  # save layer input
        # ── Standard pre-norm transformer block ──────────────────────────
        h = self.attn(self.norm1(x)) + x  # attention + residual
        h = self.ff(self.norm2(h)) + h  # FFN     + residual
        # ── Cross-layer Adaptive Fusion ──────────────────────────────────
        gate = torch.sigmoid(self.caf_gate(torch.cat([h, x_in], dim=-1)))  # [B, N, dim]
        return gate * h + (1.0 - gate) * x_in  # weighted fusion


class CAFTransformer(nn.Module):
    """Stack of depth CAFBlocks followed by a final LayerNorm."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CAFBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ── Full model ────────────────────────────────────────────────────────────


class SpectralFormer(nn.Module):
    """
    Patch-wise SpectralFormer for hyperspectral image classification.

    Accepts pre-processed patches: [B, 1, num_pca, patch_size, patch_size]
    Returns                       : [B, num_classes]

    The model interface mirrors your existing Prompt4HSI wrapper so that
    train_model() / evaluate_model() work with zero modifications:
        forward(x, y=None)
            → (ce_loss, logits, loss_dict)   when y is provided  (training)
            → (None,    logits, {})           when y is None       (inference)

    loss_dict keys: "loss_cls", "loss_rec", "loss_con"
    (loss_rec and loss_con are zero tensors; kept for API compatibility.)
    """

    def __init__(
        self,
        num_pca: int = NUM_PCA,
        patch_size: int = PATCH_SIZE,
        band_patches: int = BAND_PATCHES,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        heads: int = HEADS,
        dim_head: int = DIM_HEAD,
        mlp_dim: int = MLP_DIM,
        dropout: float = DROPOUT,
        emb_dropout: float = EMB_DROPOUT,
    ):
        super().__init__()
        self.band_patches = band_patches
        self.patch_size = patch_size

        # ── Spectral tokenisation geometry ────────────────────────────────
        # Pad spectral dim to nearest multiple of band_patches
        # e.g.  50 → 51  (17 groups of 3)
        self.num_pca_pad = ((num_pca + band_patches - 1) // band_patches) * band_patches
        self.num_tokens = self.num_pca_pad // band_patches  # 17
        self.token_dim = band_patches * patch_size * patch_size  # 3×81=243

        # ── Patch (token) embedding: LN → Linear → LN ─────────────────────
        # Matches the official repo's patch_to_embedding with spectral normalisation
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── [CLS] token and learnable positional embedding ────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens + 1, embed_dim))
        self.emb_drop = nn.Dropout(emb_dropout)

        # ── CAF Transformer encoder ────────────────────────────────────────
        self.transformer = CAFTransformer(
            embed_dim, depth, heads, dim_head, mlp_dim, dropout
        )

        # ── Classification head ────────────────────────────────────────────
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        # ── Loss function (standard CE, no masking) ────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        self._init_weights()

    # ── Weight initialisation (ViT standard) ─────────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Input pre-processing ──────────────────────────────────────────────
    def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, 1, C, H, W]  →  [B, num_tokens, token_dim]

        Step 1: squeeze channel dim     → [B, C, H, W]
        Step 2: flatten spatial dims    → [B, C, H*W]
        Step 3: zero-pad spectral dim   → [B, C_pad, H*W]
        Step 4: reshape into groups     → [B, num_tokens, band_patches*H*W]
        """
        B, _, C, H, W = x.shape
        x = x.squeeze(1)  # [B, C, H, W]
        x = x.reshape(B, C, H * W)  # [B, C, 81]

        # Zero-pad spectral dim to multiple of band_patches
        if C < self.num_pca_pad:
            pad = torch.zeros(
                B, self.num_pca_pad - C, H * W, dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, pad], dim=1)  # [B, C_pad, 81]

        # Group consecutive bands into spectral tokens
        x = x.reshape(B, self.num_tokens, self.token_dim)  # [B, 17, 243]
        return x

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ):
        """
        Args
        ----
        x : [B, 1, num_pca, H, W]  — pre-processed patch tensor
        y : [B]                     — ground-truth labels (optional)

        Returns
        -------
        (ce_loss | None,  logits [B, C],  loss_dict)

        loss_dict = {
            "loss_cls"  : cross-entropy loss (scalar tensor),
            "loss_rec"  : zero tensor (API compatibility),
            "loss_con"  : zero tensor (API compatibility),
        }
        When y is None (inference), ce_loss is None and loss_dict is empty.
        """
        # ── Tokenise & embed ──────────────────────────────────────────────
        tokens = self._tokenise(x)  # [B, 17, 243]
        tokens = self.patch_embed(tokens)  # [B, 17, 64]

        # ── Prepend [CLS] token ───────────────────────────────────────────
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 64]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 18, 64]

        # ── Add positional embedding ──────────────────────────────────────
        tokens = self.emb_drop(tokens + self.pos_embed)

        # ── CAF Transformer encoder ────────────────────────────────────────
        tokens = self.transformer(tokens)  # [B, 18, 64]

        # ── Classification from [CLS] token ──────────────────────────────
        cls_out = tokens[:, 0]  # [B, 64]
        logits = self.mlp_head(cls_out)  # [B, 16]

        # ── Loss computation ───────────────────────────────────────────────
        if y is not None:
            ce_loss = self.criterion(logits, y)
            zero = torch.zeros(1, device=x.device, dtype=ce_loss.dtype)
            loss_dict = {
                "loss_cls": ce_loss,
                "loss_rec": zero,
                "loss_con": zero,
            }
            return ce_loss, logits, loss_dict

        return None, logits, {}


def build_spectralformer() -> SpectralFormer:
    model = SpectralFormer()
    total = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpectralFormer built ✓")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {train_:,}")
    print(
        f"  Spectral tokens  : {model.num_tokens}  "
        f"(band_patches={model.band_patches}, "
        f"padded_bands={model.num_pca_pad})"
    )
    print(f"  Token dim        : {model.token_dim}  →  embed_dim={EMBED_DIM}")
    return model

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — Early Stopping
# Identical implementation to your existing notebook.
# ═══════════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """
    Monitors validation loss and stops training when no improvement
    is seen for `patience` consecutive epochs.

    Parameters
    ----------
    patience  : epochs to wait after last improvement
    delta     : minimum improvement threshold
    path      : where to save the best model checkpoint
    """

    def __init__(
        self,
        patience: int = 30,
        verbose: bool = True,
        delta: float = 1e-4,
        path: str = "best.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"  EarlyStopping: {self.counter}/{self.patience} "
                    f"(no improvement since {self.val_loss_min:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss: float, model: nn.Module):
        if self.verbose:
            self.trace_func(
                f"  ★ Val-loss {self.val_loss_min:.6f} → {val_loss:.6f}. Saving model."
            )
        # Unwrap DataParallel if needed
        state = (
            model.module if isinstance(model, nn.DataParallel) else model
        ).state_dict()
        torch.save({"model_state_dict": state, "val_loss": val_loss}, self.path)
        self.val_loss_min = val_loss

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — Training Loop
#
# Adapted from your existing train_model() but specific to SpectralFormer:
#   • Standard Cross-Entropy Loss only (no auxiliary losses)
#   • Adam optimiser with lr=5e-4, weight_decay=5e-3
#   • CosineAnnealingLR scheduler (T_max=EPOCHS)
#   • Mixed-precision (AMP) on CUDA
#   • Resume-from-checkpoint support
#   • Gradient clipping (max_norm=1.0)
# ═══════════════════════════════════════════════════════════════════════════


def get_gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return (
        f"{torch.cuda.get_device_name(0)} | "
        f"{alloc:.1f}/{reserv:.1f}/{total:.1f} GB (alloc/res/total)"
    )


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    model_tag: str = "spectralformer_IP",
) -> dict:
    """
    Full training loop for SpectralFormer.

    Returns
    -------
    history : dict with keys
        train_loss, test_loss, train_acc, test_acc, epoch_times
    """
    print(f"\n{'=' * 62}")
    print(f"  TRAINING  :  {model_tag.upper()}")
    print(f"  Device    :  {get_gpu_info()}")
    print(f"  Epochs    :  {epochs}  |  Patience : {patience}")
    print(f"{'=' * 62}")

    # ── Move model to device ──────────────────────────────────────────────
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and not isinstance(model, nn.DataParallel)
    ):
        print(f"  🚀 DataParallel: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    params = (
        model.module.parameters()
        if isinstance(model, nn.DataParallel)
        else model.parameters()
    )
    optimizer = Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── AMP scaler ────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if MIXED_PREC else None
    if scaler:
        print("  ⚡ Mixed-precision (AMP) enabled")

    # ── Paths ─────────────────────────────────────────────────────────────
    best_path = os.path.join(MODEL_DIR, f"{model_tag}_best.pth")
    ckpt_path = os.path.join(MODEL_DIR, f"{model_tag}_checkpoint.pth")

    early_stop = EarlyStopping(patience=patience, path=best_path, verbose=True)

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_times": [],
    }

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_test_loss = float("inf")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        m = model.module if isinstance(model, nn.DataParallel) else model
        m.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_test_loss = ckpt.get("best_test_loss", float("inf"))
        if "history" in ckpt:
            history = ckpt["history"]
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(
            f"  🔄 Resumed from epoch {start_epoch}  "
            f"(best loss so far: {best_test_loss:.6f})"
        )

    # ═════════════════════════════════════════════════════════════════════
    total_t0 = time.time()

    for epoch in range(start_epoch, epochs):
        ep_t0 = time.time()

        # ── TRAIN ─────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    ce_loss, logits, _ = model(data, target)
                if isinstance(model, nn.DataParallel):
                    ce_loss = ce_loss.mean()
                scaler.scale(ce_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                ce_loss, logits, _ = model(data, target)
                if isinstance(model, nn.DataParallel):
                    ce_loss = ce_loss.mean()
                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            tr_loss += ce_loss.detach().item()
            preds = logits.detach().argmax(dim=1)
            tr_correct += preds.eq(target).sum().item()
            tr_total += target.size(0)

            del data, target, ce_loss, logits, preds

            if batch_idx % 20 == 0:
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"loss: {tr_loss / (batch_idx + 1):.5f}",
                    end="\r",
                )

        scheduler.step()

        # ── EVALUATE ──────────────────────────────────────────────────────
        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        _, logits, _ = model(data)
                else:
                    _, logits, _ = model(data)

                # Compute CE loss for monitoring (does not affect weights)
                te_loss += F.cross_entropy(logits, target).item()
                preds = logits.argmax(dim=1)
                te_correct += preds.eq(target).sum().item()
                te_total += target.size(0)

                del data, target, logits, preds

        # ── Statistics ────────────────────────────────────────────────────
        avg_tr_loss = tr_loss / len(train_loader)
        avg_te_loss = te_loss / len(test_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        te_acc = 100.0 * te_correct / te_total
        ep_time = time.time() - ep_t0

        history["train_loss"].append(avg_tr_loss)
        history["test_loss"].append(avg_te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["epoch_times"].append(ep_time)

        print(
            f"  Ep {epoch + 1:3d}/{epochs} | "
            f"Tr-Loss: {avg_tr_loss:.5f}  Tr-Acc: {tr_acc:.2f}% | "
            f"Te-Loss: {avg_te_loss:.5f}  Te-Acc: {te_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{ep_time:.1f}s"
        )

        # ── Save best & checkpoint ─────────────────────────────────────────
        m = model.module if isinstance(model, nn.DataParallel) else model

        if avg_te_loss < best_test_loss:
            best_test_loss = avg_te_loss

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_test_loss": best_test_loss,
        }
        if scaler:
            ckpt_data["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt_data, ckpt_path)

        early_stop(avg_te_loss, m)
        if early_stop.early_stop:
            print(f"\n  🛑 Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % CACHE_FREQ == 0:
            clear_cache()

    # ═════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_t0
    print(f"\n  Training finished in {total_time / 60:.1f} min")
    print(f"  Best test loss : {early_stop.val_loss_min:.6f}")

    # Remove resume checkpoint after successful run
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    clear_cache()
    return history

# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — Evaluation Loop
#
# Computes the three metrics required for your IEEE TGRS submission:
#   OA  — Overall Accuracy
#   AA  — Average (per-class) Accuracy
#   K   — Cohen's Kappa Coefficient
# Plus a full per-class breakdown and confusion matrix.
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    model_tag: str = "spectralformer_IP",
) -> dict:
    """
    Full evaluation of a trained SpectralFormer.

    Returns
    -------
    results : dict containing OA, AA, Kappa, per-class accuracies,
              confusion matrix, and timing info.
    """
    print(f"\n{'=' * 62}")
    print(f"  EVALUATION  :  {model_tag.upper()}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"{'=' * 62}")

    model.eval()
    all_preds, all_targets = [], []
    t0 = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    _, logits, _ = model(data)
            else:
                _, logits, _ = model(data)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            del data, target, logits, preds

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    infer_time = time.time() - t0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── OA — Overall Accuracy ─────────────────────────────────────────────
    OA = accuracy_score(all_targets, all_preds)

    # ── AA — Average Per-Class Accuracy ───────────────────────────────────
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc.append(
                float((all_preds[mask] == c).sum()) / float(mask.sum())
            )
        else:
            per_class_acc.append(0.0)
    AA = float(np.mean(per_class_acc))

    # ── K — Cohen's Kappa ─────────────────────────────────────────────────
    K = float(cohen_kappa_score(all_targets, all_preds))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_targets, all_preds)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Overall Accuracy    (OA)  : {OA * 100:7.4f} %    │")
    print(f"  │  Average Accuracy    (AA)  : {AA * 100:7.4f} %    │")
    print(f"  │  Cohen's Kappa       (K)   : {K:10.6f}    │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"\n  Per-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        bar = "█" * int(acc * 30)
        print(f"  {i + 1:2d}. {name:<35s} {acc * 100:6.2f}%  {bar}")

    print(
        f"\n  Inference time : {infer_time:.2f}s  "
        f"({infer_time * 1000 / len(all_targets):.2f} ms/sample)"
    )

    # ── Save results to JSON ──────────────────────────────────────────────
    results = {
        "model_tag": model_tag,
        "OA": float(OA),
        "AA": float(AA),
        "Kappa": float(K),
        "per_class_accuracy": per_class_acc,
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(all_targets),
        "inference_time_s": infer_time,
        "ms_per_sample": infer_time * 1000 / len(all_targets),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(RESULTS_DIR, f"{model_tag}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    return results, cm, per_class_acc

# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — Visualisation Utilities
# ═══════════════════════════════════════════════════════════════════════════


def plot_training_history(history: dict, model_tag: str):
    """Loss and accuracy curves across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    ax.plot(epochs, history["test_loss"], "r-", lw=2, label="Test Loss")
    ax.set_title("Cross-Entropy Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-", lw=2, label="Train Acc")
    ax.plot(epochs, history["test_acc"], "r-", lw=2, label="Test Acc")
    ax.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"SpectralFormer Training History  —  {model_tag}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_tag}_training_history.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Training history saved → {path}")


def plot_confusion_matrix(cm: np.ndarray, model_tag: str):
    """Normalised confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    labels = [str(i + 1) for i in range(NUM_CLASSES)]

    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.3,
            ax=ax,
            cbar_kws={"label": title},
        )
        ax.set_title(f"Confusion Matrix — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    plt.suptitle(f"SpectralFormer  —  {model_tag}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_tag}_confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {path}")


def plot_per_class_accuracy(per_class_acc: list, model_tag: str):
    """Horizontal bar chart of per-class accuracy."""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(NUM_CLASSES)
    colors = [
        "#2196F3" if a >= 0.80 else "#FF9800" if a >= 0.60 else "#F44336"
        for a in per_class_acc
    ]

    bars = ax.barh(
        y_pos,
        [a * 100 for a in per_class_acc],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )

    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{acc * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{i + 1}. {n}" for i, n in enumerate(CLASS_NAMES)], fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 110)
    ax.axvline(
        np.mean(per_class_acc) * 100,
        color="green",
        linestyle="--",
        lw=1.5,
        label=f"AA = {np.mean(per_class_acc) * 100:.2f}%",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"Per-Class Accuracy  —  SpectralFormer ({model_tag})",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_tag}_per_class_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Per-class accuracy saved → {path}")

# ═══════════════════════════════════════════════════════════════════════════
# CELL 9 — Main Execution
# Run this cell to train and evaluate SpectralFormer on Indian Pines.
# All outputs (model, results JSON, plots) are written to /kaggle/working/.
# ═══════════════════════════════════════════════════════════════════════════


def main():
    MODEL_TAG = f"spectralformer_{DATASET_ABBR}_bp{BAND_PATCHES}"

    print("=" * 62)
    print("  SpectralFormer — Patch-Wise CAF Mode")
    print(f"  Dataset   : Indian Pines ({DATASET_ABBR})")
    print(f"  Classes   : {NUM_CLASSES}")
    print(f"  Input     : [B, 1, {NUM_PCA}, {PATCH_SIZE}, {PATCH_SIZE}]")
    print(f"  Model tag : {MODEL_TAG}")
    print("=" * 62)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading DataLoaders …")
    train_loader, test_loader = get_dataloaders(
        dataset_abbr=DATASET_ABBR,
        pca_components=NUM_PCA,
        batch_size=BATCH_SIZE,
    )

    # ── Step 2: Build model ───────────────────────────────────────────────
    print("\n[2/4] Building SpectralFormer …")
    model = build_spectralformer()

    # ── Step 3: Train ─────────────────────────────────────────────────────
    print("\n[3/4] Training …")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        model_tag=MODEL_TAG,
    )

    # ── Step 4: Load best checkpoint and evaluate ─────────────────────────
    print("\n[4/4] Evaluating best model …")
    best_path = os.path.join(MODEL_DIR, f"{MODEL_TAG}_best.pth")

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        state = ckpt["model_state_dict"]
        # Strip DataParallel prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Rebuild a fresh model and load weights
        best_model = build_spectralformer().to(DEVICE)
        best_model.load_state_dict(state)
        print("  ✓ Best model weights loaded")
    else:
        print("  ⚠ best.pth not found — evaluating current model weights")
        best_model = model

    results, cm, per_class_acc = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        device=DEVICE,
        model_tag=MODEL_TAG,
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[Plots] Generating visualisations …")
    plot_training_history(history, MODEL_TAG)
    plot_confusion_matrix(cm, MODEL_TAG)
    plot_per_class_accuracy(per_class_acc, MODEL_TAG)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  IEEE TGRS Metrics  —  SpectralFormer (Indian Pines)")
    print(f"{'═' * 62}")
    print(f"  OA  (Overall Accuracy)        : {results['OA'] * 100:.4f} %")
    print(f"  AA  (Average Accuracy)        : {results['AA'] * 100:.4f} %")
    print(f"  K   (Cohen's Kappa)           : {results['Kappa']:.6f}")
    print(f"{'═' * 62}")
    print(f"  Results JSON → {RESULTS_DIR}")
    print(f"  Plots        → {PLOTS_DIR}")
    print(f"  Best model   → {best_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────
results = main()
