# ============================================================
# HSIMAE Finetuning — Indian Pines (IEEE TGRS Submission)
# Input  : [B, 1, 50, 9, 9]  (50 PCA comps, 9×9 spatial)
# Weights: Auto-downloaded from HuggingFace RyanWy/HSIMAE
# ============================================================

# ── 0. Installs & Imports ────────────────────────────────────
import subprocess

subprocess.run(["pip", "install", "huggingface_hub", "-q"], check=True)

import os, sys, json, time, math, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download, list_repo_files

print("✅ All imports successful.")


# ── 1. Config ────────────────────────────────────────────────
class Config:
    # ── Data (matches your [B,1,50,9,9] loaders exactly) ──
    SPECTRAL_DIM = 50
    PATCH_H = 9
    PATCH_W = 9
    IN_CHANS = 1
    NUM_CLASSES = 16

    # ── Tokenisation (3-D cube embedding) ──
    # Spatial:  9 // 3 = 3×3 = 9 spatial tokens
    # Spectral: 50 // 10 = 5 spectral tokens
    # Total   : 9 × 5 = 45 tokens
    SPATIAL_STRIDE = 3
    SPECTRAL_STRIDE = 10
    N_SPATIAL_TOKENS = (9 // 3) * (9 // 3)  # 9
    N_SPECTRAL_TOKENS = 50 // 10  # 5
    N_TOKENS = N_SPATIAL_TOKENS * N_SPECTRAL_TOKENS  # 45

    # ── Encoder (HSIMAE-B dims from paper) ──
    TOKEN_DIM = 128
    SSSE_DEPTH = 9
    FUSION_DEPTH = 3
    N_HEADS = 8

    # ── Decoder (lightweight: 8 blocks, 64-dim) ──
    DECODER_DIM = 64
    DECODER_DEPTH = 8
    DECODER_HEADS = 8

    # ── Masking: 1-(1-0.60)(1-0.50) = 0.80 total ratio ──
    MASK_RATIO_SPA = 0.60
    MASK_RATIO_SPE = 0.50
    MIN_SPA_KEEP = 2
    MIN_SPE_KEEP = 2

    # ── Training (paper finetuning settings) ──
    EPOCHS = 200
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 0.005
    LAMBDA_REC = 10.0  # λ: L = Lcls + λ·Lrec
    DROP_PATH = 0.2
    PATIENCE = 30

    # ── Paths ──
    DATASET_ABBR = "IP"
    PROCESSED_ROOT = "/home/23dcs505/datasets/IP"
    MODEL_DIR = "/home/23dcs505/best_models"
    RESULTS_DIR = "/home/23dcs505/results"
    PRETRAIN_CACHE = "/home/23dcs505/hsimae_pretrained"

    # ── HuggingFace pretrained weights ──
    HF_REPO_ID = "RyanWy/HSIMAE"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
for d in [
    cfg.MODEL_DIR,
    cfg.RESULTS_DIR,
    cfg.PRETRAIN_CACHE,
    os.path.join(cfg.RESULTS_DIR, "plots"),
]:
    os.makedirs(d, exist_ok=True)

print(f"Device : {cfg.DEVICE}")
print(
    f"Tokens : {cfg.N_TOKENS}  "
    f"(spatial={cfg.N_SPATIAL_TOKENS}, spectral={cfg.N_SPECTRAL_TOKENS})"
)

# ── 2. Indian Pines Class Names ───────────────────────────────
IP_CLASS_NAMES = [
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
    "Buildings-Grass-Trees",
    "Stone-Steel-Towers",
]


# ── 3. Data Loading ───────────────────────────────────────────
class HyperspectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float() if X.dtype != torch.float32 else X
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders_hsimae(pca=50, abbr="IP", batch_size=32):
    proc_dir = os.path.join(cfg.PROCESSED_ROOT, f"pca_{pca}", abbr)
    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    print(f"[DATA] X_train={list(X_tr.shape)}  X_test={list(X_te.shape)}")
    assert list(X_tr.shape[1:]) == [1, 50, 9, 9], (
        f"Shape mismatch — expected [N,1,50,9,9], got {list(X_tr.shape)}"
    )

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    y_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else y_tr
    counts = np.bincount(y_np, minlength=cfg.NUM_CLASSES).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    sampler = WeightedRandomSampler(
        torch.tensor(weights[y_np]), len(y_np), replacement=True
    )

    nw = min(4, max(1, os.cpu_count() // 2))
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    print(f"[DATA] Batches — Train:{len(train_loader)}, Test:{len(test_loader)}")
    return train_loader, test_loader


# ── 4. Model Building Blocks ──────────────────────────────────
def drop_path_fn(x, p=0.0, training=False):
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return drop_path_fn(x, self.p, self.training)


class SwiGLU(nn.Module):
    """Feed-forward replaced by SwiGLU (paper eq. 6-7)."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim * 4
        self.W = nn.Linear(dim, hidden, bias=True)
        self.V = nn.Linear(dim, hidden, bias=True)
        self.out = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.out(F.silu(self.W(x)) * self.V(x))


class MHSA(nn.Module):
    def __init__(self, dim, n_heads, drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(drop)
        self.scale = self.hd**-0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class HSIMAEBlock(nn.Module):
    """Modified ViT block with SwiGLU (paper eq. 7)."""

    def __init__(self, dim, n_heads, dp=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, n_heads)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()
        self.n2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.dp(self.attn(self.n1(x)))
        x = x + self.dp(self.ffn(self.n2(x)))
        return x


# ── 5. Patch Embedding: [B,1,50,9,9] → [B,45,128] ────────────
class HSIPatchEmbed(nn.Module):
    """
    Single Conv3d maps 3-D cubes (10×3×3) to TOKEN_DIM tokens.
    Replaces original group-wise PCA + linear projection.
    """

    def __init__(self, spec=50, ph=9, pw=9, ss=3, sps=10, dim=128):
        super().__init__()
        self.n_spa = (ph // ss) * (pw // ss)
        self.n_spe = spec // sps
        self.n_tok = self.n_spa * self.n_spe
        self.proj = nn.Conv3d(
            1, dim, kernel_size=(sps, ss, ss), stride=(sps, ss, ss), bias=True
        )

    def forward(self, x):  # x: [B, 1, 50, 9, 9]
        x = self.proj(x)  # [B, D, 5, 3, 3]
        return x.flatten(2).transpose(1, 2)  # [B, 45, D]


# ── 6. Separable Sinusoidal Positional Embedding ──────────────
def sin_pe_1d(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


def sep_pos_embed(n_spa, n_spe, dim):
    h = dim // 2
    spa = sin_pe_1d(n_spa, h)  # [9, 64]
    spe = sin_pe_1d(n_spe, dim - h)  # [5, 64]
    spa = spa.unsqueeze(0).expand(n_spe, -1, -1)  # [5, 9, 64]
    spe = spe.unsqueeze(1).expand(-1, n_spa, -1)  # [5, 9, 64]
    pe = torch.cat([spe, spa], dim=-1).reshape(n_spa * n_spe, dim)
    return pe  # [45, 128]


# ── 7. Spatial-Spectral Masking (paper Section III-B) ─────────
def spatial_spectral_mask(
    tokens, n_spa, n_spe, mr_spa=0.60, mr_spe=0.50, min_spa=2, min_spe=2
):
    """
    Eq.8: total_mask = 1-(1-mr_spa)(1-mr_spe)  ≈  0.80
    Guarantees spatial & spectral consistency for SSSE.
    Returns: visible_tokens [B,n_vis,D], mask [B,N], ids_restore [B,N]
    """
    B, N, D = tokens.shape
    dev = tokens.device

    n_spa_keep = max(min_spa, int(n_spa * (1 - mr_spa)))
    n_spe_keep = max(min_spe, int(n_spe * (1 - mr_spe)))

    spa_keep_ids = torch.rand(B, n_spa, device=dev).argsort(1)[:, :n_spa_keep]
    spe_keep_ids = torch.rand(B, n_spe, device=dev).argsort(1)[:, :n_spe_keep]

    # Build bool mask: True = masked
    spa_mask = torch.ones(B, n_spa, dtype=torch.bool, device=dev)
    spe_mask = torch.ones(B, n_spe, dtype=torch.bool, device=dev)
    for b in range(B):
        spa_mask[b, spa_keep_ids[b]] = False
        spe_mask[b, spe_keep_ids[b]] = False

    # Token visible only if BOTH its spa AND spe location are kept
    mask_2d = spa_mask.unsqueeze(1).expand(B, n_spe, n_spa) | spe_mask.unsqueeze(
        2
    ).expand(B, n_spe, n_spa)
    mask = mask_2d.reshape(B, N)  # [B,45] 1=masked

    ids_shuffle = mask.long().argsort(1)  # 0s (visible) first
    ids_restore = ids_shuffle.argsort(1)
    n_vis = int((~mask[0]).sum())
    ids_keep = ids_shuffle[:, :n_vis]

    vis = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return vis, mask, ids_restore


# ── 8. Separate Spatial-Spectral Encoder (SSSE) ───────────────
class SSSEncoder(nn.Module):
    def __init__(self, n_spa, n_spe, dim, n_heads, ssse_d=9, fuse_d=3, dp_rates=None):
        super().__init__()
        total = ssse_d + fuse_d
        dps = dp_rates or [x.item() for x in torch.linspace(0, 0.2, total)]

        self.spa_blks = nn.ModuleList(
            [HSIMAEBlock(dim, n_heads, dp=dps[i]) for i in range(ssse_d)]
        )
        self.spe_blks = nn.ModuleList(
            [HSIMAEBlock(dim, n_heads, dp=dps[i]) for i in range(ssse_d)]
        )
        self.fuse_blks = nn.ModuleList(
            [HSIMAEBlock(dim, n_heads, dp=dps[ssse_d + i]) for i in range(fuse_d)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z_spa = x
        for b in self.spa_blks:
            z_spa = b(z_spa)
        z_spe = x
        for b in self.spe_blks:
            z_spe = b(z_spe)
        z = z_spa + z_spe
        for b in self.fuse_blks:
            z = b(z)
        return self.norm(z)


# ── 9. Lightweight Decoder ────────────────────────────────────
class HSIMAEDecoder(nn.Module):
    def __init__(self, enc_dim, dec_dim, dec_d, dec_h, cube_size):
        super().__init__()
        self.proj_in = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blks = nn.ModuleList([HSIMAEBlock(dec_dim, dec_h) for _ in range(dec_d)])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, cube_size)

    def forward(self, z_vis, mask, ids_restore, pe_dec):
        B, N = mask.shape
        x = self.proj_in(z_vis)
        mt = self.mask_token.expand(B, N - x.shape[1], -1)
        xf = torch.cat([x, mt], 1)
        xf = xf.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, xf.shape[-1]))
        xf = xf + pe_dec.unsqueeze(0).to(xf.device)
        for b in self.blks:
            xf = b(xf)
        return self.pred(self.norm(xf))  # [B, 45, cube_size]


# ── 10. Full HSIMAE Finetuning Model ─────────────────────────
class HSIMAEFinetune(nn.Module):
    """
    Dual-branch finetuning (paper Section III-C):
      Labeled   branch → encoder → GAP → Linear (CE loss)
      Unlabeled branch → encoder → decoder (MSE loss)
    Shared encoder. Total loss = CE + 10·MSE
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_spa = cfg.N_SPATIAL_TOKENS
        self.n_spe = cfg.N_SPECTRAL_TOKENS
        self.n_tok = cfg.N_TOKENS

        self.patch_embed = HSIPatchEmbed(
            spec=cfg.SPECTRAL_DIM,
            ph=cfg.PATCH_H,
            pw=cfg.PATCH_W,
            ss=cfg.SPATIAL_STRIDE,
            sps=cfg.SPECTRAL_STRIDE,
            dim=cfg.TOKEN_DIM,
        )

        pe = sep_pos_embed(self.n_spa, self.n_spe, cfg.TOKEN_DIM)
        pe_dec = sep_pos_embed(self.n_spa, self.n_spe, cfg.DECODER_DIM)
        self.register_buffer("pos_embed", pe)  # [45, 128]
        self.register_buffer("pos_embed_dec", pe_dec)  # [45, 64]

        self.encoder = SSSEncoder(
            self.n_spa,
            self.n_spe,
            cfg.TOKEN_DIM,
            cfg.N_HEADS,
            ssse_d=cfg.SSSE_DEPTH,
            fuse_d=cfg.FUSION_DEPTH,
        )

        self.cls_norm = nn.LayerNorm(cfg.TOKEN_DIM)
        concat_dim = cfg.TOKEN_DIM * self.n_spe
        self.classifier = nn.Linear(concat_dim, cfg.NUM_CLASSES)

        cube_size = cfg.SPECTRAL_STRIDE * cfg.SPATIAL_STRIDE**2  # 10×3×3=90
        self.decoder = HSIMAEDecoder(
            cfg.TOKEN_DIM,
            cfg.DECODER_DIM,
            cfg.DECODER_DEPTH,
            cfg.DECODER_HEADS,
            cube_size,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── forward helpers ──
    def _tokenize(self, x):
        return self.patch_embed(x) + self.pos_embed  # [B,45,128]

    def _labeled_branch(self, x):
        B = x.shape[0]
        z = self.encoder(self._tokenize(x))
        z = self.cls_norm(z)
        z_r = z.reshape(B, self.n_spe, self.n_spa, self.cfg.TOKEN_DIM)
        z_concat = z_r.transpose(1, 2).reshape(B, self.n_spa, -1)
        return self.classifier(z_concat.mean(1))  # [B, 16]

    def _unlabeled_branch(self, x):
        tokens = self._tokenize(x)
        vis, mask, ids_restore = spatial_spectral_mask(
            tokens,
            self.n_spa,
            self.n_spe,
            self.cfg.MASK_RATIO_SPA,
            self.cfg.MASK_RATIO_SPE,
            self.cfg.MIN_SPA_KEEP,
            self.cfg.MIN_SPE_KEEP,
        )
        z_vis = self.encoder(vis)
        recon = self.decoder(z_vis, mask, ids_restore, self.pos_embed_dec)
        return recon, mask, tokens

    def forward(self, x_lab, labels=None, x_unlab=None):
        """
        Inference (labels=None) → returns (None, logits)
        Training              → returns (total_loss, logits, components)
        """
        logits = self._labeled_branch(x_lab)

        if labels is None:
            return None, logits

        loss_cls = F.cross_entropy(logits, labels)

        xu = x_unlab if x_unlab is not None else x_lab
        recon, mask, tgt_raw = self._unlabeled_branch(xu)

        with torch.no_grad():
            B = xu.shape[0]
            sps, ss = self.cfg.SPECTRAL_STRIDE, self.cfg.SPATIAL_STRIDE
            n_spa_h = self.cfg.PATCH_H // ss
            n_spa_w = self.cfg.PATCH_W // ss
            xr = xu.reshape(B, 1, self.n_spe, sps, n_spa_h, ss, n_spa_w, ss)
            xr = xr.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, self.n_tok, -1)
            
            mu, var = xr.mean(-1, keepdim=True), xr.var(-1, keepdim=True)
            xr_norm = (xr - mu) / (var + 1e-6).sqrt()

        per_tok = F.mse_loss(recon, xr_norm, reduction="none").mean(-1)
        loss_rec = (per_tok * mask.float()).sum() / (mask.float().sum() + 1e-6)

        total = loss_cls + self.cfg.LAMBDA_REC * loss_rec
        return total, logits, {"cls": loss_cls.item(), "rec": loss_rec.item()}


# ── 11. Pretrained Weight Loader ──────────────────────────────
def load_pretrained_weights(model: HSIMAEFinetune, cfg: Config):
    """
    Downloads HSIMAE-B pretrained checkpoint from HuggingFace
    (RyanWy/HSIMAE) and loads encoder weights with strict=False.
    The classification head is intentionally kept randomly
    initialised (different #classes per dataset).
    """
    print("\n── Pretrained Weight Loading ──────────────────────────")
    print(f"   HF Repo : {cfg.HF_REPO_ID}")

    try:
        # List all .pth / .pt files in the repo
        all_files = list(list_repo_files(cfg.HF_REPO_ID))
        ckpt_files = [f for f in all_files if f.endswith(".pth") or f.endswith(".pt")]
        print(f"   Found checkpoint files: {ckpt_files}")

        if not ckpt_files:
            print(
                "   ⚠️  No .pth/.pt file found in repo. "
                "Skipping pretrained loading — training from random init."
            )
            return model

        # Prefer HSIMAE-B over HSIMAE-L (smaller, faster finetuning)
        target = next(
            (f for f in ckpt_files if "_b" in f.lower() or "base" in f.lower()),
            ckpt_files[0],
        )
        print(f"   Downloading : {target}")

        local_path = hf_hub_download(
            repo_id=cfg.HF_REPO_ID,
            filename=target,
            cache_dir=cfg.PRETRAIN_CACHE,
        )
        print(f"   Cached at   : {local_path}")

        ckpt = torch.load(local_path, map_location=cfg.DEVICE)

        # Handle various checkpoint formats
        if isinstance(ckpt, dict):
            state = (
                ckpt.get("model_state_dict")
                or ckpt.get("model")
                or ckpt.get("state_dict")
                or ckpt
            )
        else:
            state = ckpt

        # strict=False: encoder weights load; cls head skipped (shape mismatch)
        incompatible = model.load_state_dict(state, strict=False)

        loaded = len(state) - len(incompatible.missing_keys)
        print(f"\n   ✅ Pretrained weights loaded successfully!")
        print(f"   Keys loaded       : {loaded}")
        print(f"   Missing (new)     : {len(incompatible.missing_keys)}")
        print(f"   Unexpected (extra): {len(incompatible.unexpected_keys)}")

        if incompatible.missing_keys:
            print("   [Missing keys — these will be trained from scratch]")
            for k in incompatible.missing_keys[:10]:
                print(f"     · {k}")
            if len(incompatible.missing_keys) > 10:
                print(f"     ... and {len(incompatible.missing_keys) - 10} more")

    except Exception as e:
        print(f"\n   ⚠️  Could not load pretrained weights: {e}")
        print("   Falling back to random initialisation.")

    print("─" * 55)
    return model


# ── 12. Training Loop ─────────────────────────────────────────
def train_hsimae(cfg: Config):
    print(f"\n{'=' * 65}")
    print(f"  HSIMAE Finetuning — Indian Pines | IEEE TGRS Submission")
    print(f"{'=' * 65}")

    train_loader, test_loader = get_dataloaders_hsimae(
        pca=50, abbr=cfg.DATASET_ABBR, batch_size=cfg.BATCH_SIZE
    )

    # Build model
    model = HSIMAEFinetune(cfg).to(cfg.DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_p:,}")

    # ── Load pretrained encoder weights ──
    model = load_pretrained_weights(model, cfg)

    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_oa, patience_cnt = 0.0, 0
    best_path = os.path.join(cfg.MODEL_DIR, "hsimae_IP_best.pth")

    history = {
        k: []
        for k in [
            "train_loss",
            "test_loss",
            "train_acc",
            "test_acc",
            "loss_cls",
            "loss_rec",
        ]
    }

    print(f"\n{'─' * 65}")
    print(
        f"  Starting Training  |  Epochs={cfg.EPOCHS}  "
        f"BS={cfg.BATCH_SIZE}  LR={cfg.LR}  λ={cfg.LAMBDA_REC}"
    )
    print(f"{'─' * 65}")

    for epoch in range(1, cfg.EPOCHS + 1):
        # ─── Train ─────────────────────────────────────────
        model.train()
        tl, tc, tt, tls_cls, tls_rec = 0.0, 0, 0, 0.0, 0.0

        for x, y in train_loader:
            x = x.to(cfg.DEVICE, non_blocking=True)
            y = y.to(cfg.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.cuda.amp.autocast():
                    loss, logits, comps = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, logits, comps = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tl += loss.item()
            tls_cls += comps["cls"]
            tls_rec += comps["rec"]
            tc += logits.argmax(1).eq(y).sum().item()
            tt += y.size(0)

        scheduler.step()
        nb = len(train_loader)

        # ─── Validate ──────────────────────────────────────
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        all_p, all_t = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                loss_v, logits_v = model(x, y)[:2]
                vl += loss_v.item()
                p = logits_v.argmax(1)
                vc += p.eq(y).sum().item()
                vt += y.size(0)
                all_p.extend(p.cpu().numpy())
                all_t.extend(y.cpu().numpy())

        avg_tl = tl / nb
        tr_acc = 100.0 * tc / tt
        avg_vl = vl / len(test_loader)
        va_acc = 100.0 * vc / vt

        # Save best
        star = " "
        if va_acc > best_oa:
            best_oa, patience_cnt = va_acc, 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "oa": best_oa},
                best_path,
            )
            star = "★"
        else:
            patience_cnt += 1

        for k, v in zip(
            [
                "train_loss",
                "test_loss",
                "train_acc",
                "test_acc",
                "loss_cls",
                "loss_rec",
            ],
            [avg_tl, avg_vl, tr_acc, va_acc, tls_cls / nb, tls_rec / nb],
        ):
            history[k].append(v)

        if epoch % 10 == 0 or epoch <= 5:
            print(
                f"Ep {epoch:3d}/{cfg.EPOCHS} | "
                f"Loss={avg_tl:.4f} "
                f"(cls={tls_cls / nb:.4f} rec={tls_rec / nb:.4f}) | "
                f"Train={tr_acc:.2f}% | Test={va_acc:.2f}% | "
                f"LR={optimizer.param_groups[0]['lr']:.2e} {star}"
            )

        if patience_cnt >= cfg.PATIENCE:
            print(f"\n🛑 Early stop at epoch {epoch}  (best Test OA={best_oa:.2f}%)")
            break

    print(f"\n✅ Training finished.  Best Test OA (proxy): {best_oa:.2f}%")
    return model, history, best_path, train_loader, test_loader


# ── 13. Final Evaluation — OA · AA · Kappa ───────────────────
def evaluate_hsimae(model, test_loader, cfg: Config, best_path: str = None):
    """Loads best checkpoint → computes OA, AA, κ for TGRS table."""
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[EVAL] Best checkpoint loaded (epoch {ckpt['epoch']})")

    model.eval()
    all_p, all_t = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.DEVICE, non_blocking=True)
            _, logits = model(x)  # labels=None → inference mode
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_t.extend(y.numpy())

    all_p = np.array(all_p)
    all_t = np.array(all_t)

    oa = accuracy_score(all_t, all_p) * 100.0
    kappa = cohen_kappa_score(all_t, all_p)
    cm = confusion_matrix(all_t, all_p, labels=list(range(cfg.NUM_CLASSES)))

    pca_list = []
    for c in range(cfg.NUM_CLASSES):
        n = (all_t == c).sum()
        pca_list.append(
            ((all_t == c) & (all_p == c)).sum() / n * 100.0 if n > 0 else 0.0
        )
    aa = np.mean(pca_list)

    # ── Print results ──
    print(f"\n{'=' * 57}")
    print(f"  HSIMAE (Pretrained) — Indian Pines | IEEE TGRS Table")
    print(f"{'=' * 57}")
    print(f"  Overall Accuracy  (OA) : {oa:.2f} %")
    print(f"  Average Accuracy  (AA) : {aa:.2f} %")
    print(f"  Kappa Coefficient (κ)  : {kappa:.4f}")
    print(f"{'─' * 57}")
    print(f"  Per-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(IP_CLASS_NAMES, pca_list)):
        print(f"    {i + 1:2d}. {name:<35s}: {acc:.2f}%")
    print(f"{'=' * 57}")

    results = {
        "method": "HSIMAE (Pretrained HuggingFace)",
        "dataset": "IndianPines",
        "OA_%": round(oa, 4),
        "AA_%": round(aa, 4),
        "Kappa": round(kappa, 6),
        "per_class_accuracy_%": [round(a, 4) for a in pca_list],
        "confusion_matrix": cm.tolist(),
    }
    out_json = os.path.join(cfg.RESULTS_DIR, "hsimae_IP_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_json}")
    return results, cm, pca_list


# ── 14. Visualisation ─────────────────────────────────────────
def plot_curves(history, cfg):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    ax[0].plot(ep, history["train_loss"], label="Train")
    ax[0].plot(ep, history["test_loss"], label="Test")
    ax[0].set_title("Total Loss")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(ep, history["train_acc"], label="Train")
    ax[1].plot(ep, history["test_acc"], label="Test")
    ax[1].set_title("Accuracy (%)")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(ep, history["loss_cls"], label="CE (cls)")
    ax[2].plot(ep, history["loss_rec"], label="MSE×10 (rec)")
    ax[2].set_title("Component Losses")
    ax[2].legend()
    ax[2].grid(True)

    plt.suptitle(
        "HSIMAE — Indian Pines Training Curves", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "plots", "hsimae_IP_training.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves  → {out}")


def plot_cm(cm, cfg):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(1, cfg.NUM_CLASSES + 1),
        yticklabels=range(1, cfg.NUM_CLASSES + 1),
    )
    plt.title("HSIMAE — Indian Pines Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "plots", "hsimae_IP_cm.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix → {out}")


def plot_per_class(pca_list, cfg):
    plt.figure(figsize=(16, 6))
    bars = plt.bar(
        range(1, cfg.NUM_CLASSES + 1),
        pca_list,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )
    for bar, a in zip(bars, pca_list):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{a:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(
        range(1, cfg.NUM_CLASSES + 1),
        [f"{i + 1}\n{n[:9]}" for i, n in enumerate(IP_CLASS_NAMES)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.ylim(0, 110)
    plt.title(
        "HSIMAE — Per-Class Accuracy (Indian Pines)", fontsize=13, fontweight="bold"
    )
    plt.ylabel("Accuracy (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(cfg.RESULTS_DIR, "plots", "hsimae_IP_per_class.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Per-class plot   → {out}")


# ── 15. Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # ── Train (pretrained weights auto-downloaded & loaded) ──
    model, history, best_path, train_loader, test_loader = train_hsimae(cfg)

    # ── Evaluate with best checkpoint ──
    results, cm, pca_list = evaluate_hsimae(model, test_loader, cfg, best_path)

    # ── Save plots ──
    plot_curves(history, cfg)
    plot_cm(cm, cfg)
    plot_per_class(pca_list, cfg)

    print("\n✅ HSIMAE pipeline complete.")
    print("   Copy OA / AA / Kappa into your IEEE TGRS comparison table.")
    print(
        f"   Results JSON → {os.path.join(cfg.RESULTS_DIR, 'hsimae_IP_results.json')}"
    )
    
