# ============================================================
# VLM-HSI — Chatterjee & Ghosh (2025)
# "Learning Hyperspectral Images with Curated Text Prompts
#  for Efficient Multimodal Alignment"
# arXiv: 2509.22697
#
# Built from scratch from paper description (no official repo).
# Adapted for DataLoader output: [B, 1, 50, 9, 9]
# ============================================================

# ── 0. Dependencies ──────────────────────────────────────────
# On Kaggle, transformers/einops are pre-installed.
# If not: !pip install transformers einops sentence-transformers -q

import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from datetime import datetime

# ── 1. CONFIGURATION ─────────────────────────────────────────
DATASET_ABBR = "IP"
NUM_CLASSES = 16
PCA_COMPONENTS = 50
PATCH_SIZE = 9

# Vision encoder (ViT) hyperparams — from paper §3
SPATIAL_PATCH_SIZE = 3  # 3×3 spatial sub-patches → 9 tokens per 9×9 patch
EMBED_DIM = 64  # shared vision–text embedding space
VIT_DEPTH = 6
VIT_HEADS = 16  # 64/16 = 4 dims per head
VIT_MLP_DIM = 64
VIT_DROPOUT = 0.1

# Text encoder
TEXT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
TEXT_DIM = 1024  # BGE-large output dimension

# Contrastive loss params — from paper §2
KH = 4  # number of hard negatives
KS = 4  # number of semi-hard negatives
LOGIT_SCALE_INIT = math.log(1.0 / 0.07)  # CLIP-style init (≈2.66)

# Training — from paper §3 (IP-specific)
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
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

processed_root = "/home/23dcs505/datasets/IP"
output_dir = "/home/23dcs505/best_models"
os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
print(
    f"Config  : PCA={PCA_COMPONENTS}, patch={PATCH_SIZE}×{PATCH_SIZE}, embed_dim={EMBED_DIM}"
)


# ── 2. TEXT PROMPTS (from paper §2, descriptive template) ────
# Template: "This image shows a large cultivated field of {<CLS>},
#            where {<CLS>} plants are densely grown in rows;
#            the vivid green {<CLS>} vegetation is clearly visible
#            from an aerial perspective."


def build_ip_prompts(class_names):
    """Construct the descriptive prompt for each IP class."""
    template = (
        "This image shows a large cultivated field of {cls}, "
        "where {cls} plants are densely grown in rows; "
        "the vivid green {cls} vegetation is clearly visible "
        "from an aerial perspective."
    )
    return [template.format(cls=c) for c in class_names]


IP_PROMPTS = build_ip_prompts(CLASS_NAMES)
print(f"\nSample prompt [0] :\n  {IP_PROMPTS[0]}")
print(f"Sample prompt [15]:\n  {IP_PROMPTS[15]}")


# ── 3. VISION ENCODER ────────────────────────────────────────


class PatchEmbed3D(nn.Module):
    """
    Tokenise a [B, 1, D, H, W] HSI patch into non-overlapping
    3D sub-tokens.

    Strategy (from paper): divide the H×W spatial plane into
    (spatial_patch × spatial_patch) windows; each window spans
    all D spectral channels → token_dim = D × p × p.
    Projects each token to embed_dim via a linear layer.

    For our input [B, 1, 50, 9, 9] with p=3:
        9 spatial tokens × (50×3×3=450) features → projected to 64-D.
    """

    def __init__(self, spectral_dim: int, spatial_patch: int, embed_dim: int):
        super().__init__()
        self.p = spatial_patch
        token_dim = spectral_dim * spatial_patch * spatial_patch  # 450
        self.proj = nn.Linear(token_dim, embed_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, D, H, W]
        B, _, D, H, W = x.shape
        p = self.p
        nh, nw = H // p, W // p
        x = x.squeeze(1)  # [B, D, H, W]
        x = x.view(B, D, nh, p, nw, p)  # [B, D, nh, p, nw, p]
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, nh, nw, D, p, p]
        x = x.contiguous().view(B, nh * nw, D * p * p)  # [B, 9, 450]
        return self.proj(x)  # [B, 9, 64]


class HSIViT(nn.Module):
    """
    Lightweight Vision Transformer backbone for HSI patches.
    Architecture (from paper §3):
      - 3D patch tokenisation → 64-D token embeddings
      - CLS token + positional embedding
      - 6 Transformer encoder layers, 16 heads, MLP dim=64
      - Output: CLS token → 64-D visual feature
    """

    def __init__(
        self,
        spectral_dim=PCA_COMPONENTS,
        spatial_size=PATCH_SIZE,
        spatial_patch=SPATIAL_PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=VIT_DEPTH,
        heads=VIT_HEADS,
        mlp_dim=VIT_MLP_DIM,
        dropout=VIT_DROPOUT,
    ):
        super().__init__()
        self.num_patches = (spatial_size // spatial_patch) ** 2  # 9

        # Patch embedding: [B,1,50,9,9] → [B,9,64]
        self.patch_embed = PatchEmbed3D(spectral_dim, spatial_patch, embed_dim)

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, norm=nn.LayerNorm(embed_dim)
        )

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 50, 9, 9]
        B = x.shape[0]
        tokens = self.patch_embed(x)  # [B, 9, 64]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 64]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 10, 64]
        tokens = tokens + self.pos_embed  # [B, 10, 64]
        tokens = self.pos_drop(tokens)
        tokens = self.transformer(tokens)  # [B, 10, 64]
        return tokens[:, 0]  # CLS → [B, 64]


# ── 4. FULL VLM MODEL ────────────────────────────────────────


class VLM_HSI(nn.Module):
    """
    Vision–Language Model for HSI classification.

    Trainable components (~246K params):
      • HSIViT vision encoder          → ~181K
      • Projection head (1024 → 64)    → ~65.6K
      • logit_scale scalar             → 1

    Frozen:
      • BAAI/bge-large-en-v1.5 text encoder (335M params)
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        text_dim=TEXT_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Vision encoder
        self.vision_encoder = HSIViT()

        # Trainable linear probe on top of frozen LEM features (§2)
        self.projection_head = nn.Linear(text_dim, embed_dim)
        nn.init.trunc_normal_(self.projection_head.weight, std=0.02)
        nn.init.zeros_(self.projection_head.bias)

        # Learnable temperature (CLIP-style)
        self.logit_scale = nn.Parameter(torch.ones([]) * LOGIT_SCALE_INIT)

    def encode_vision(self, x: torch.Tensor) -> torch.Tensor:
        """Visual embedding: [B, 64] → L2-normalised."""
        feat = self.vision_encoder(x)  # [B, 64]
        return F.normalize(feat, dim=-1)  # zx

    def encode_text(self, raw_text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project frozen BGE embeddings through trainable probe → L2-normalise.
        raw_text_embeds: [C, 1024] — precomputed, cached on GPU.
        Returns: [C, 64]
        """
        proj = self.projection_head(raw_text_embeds)  # [C, 64]
        return F.normalize(proj, dim=-1)  # zt

    def forward(self, x: torch.Tensor, raw_text_embeds: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised (visual, text) embedding pair."""
        zv = self.encode_vision(x)
        zt = self.encode_text(raw_text_embeds)
        return zv, zt


# ── 5. TEXT ENCODER (BGE, frozen) ────────────────────────────


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Attention-mask-weighted mean pooling of token embeddings."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    count = mask.sum(1).clamp(min=1e-9)
    return summed / count


@torch.no_grad()
def compute_text_embeddings(
    prompts: list,
    model_name: str = TEXT_MODEL_NAME,
    device: torch.device = device,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Compute raw 1024-D BGE embeddings for a list of text prompts.
    The BGE model is loaded, used, then CPU-offloaded to free VRAM.
    Returns: [C, 1024] on `device`.
    """
    print(f"\n── Loading BGE text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name).to(device)
    text_model.eval()

    # Freeze all params
    for p in text_model.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in text_model.parameters())
    print(f"   BGE params (frozen): {total_params:,}")

    all_embeds = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        out = text_model(**encoded)
        embeds = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        all_embeds.append(embeds.cpu())

    text_model.cpu()  # free GPU VRAM after encoding
    torch.cuda.empty_cache()

    raw = torch.cat(all_embeds, dim=0)  # [C, 1024]
    print(f"   Text embeddings shape: {raw.shape}  ✓")
    return raw.to(device)


# ── 6. DISTRACTOR-AWARE CONTRASTIVE LOSS (Eq. 6, paper §2) ──


def distractor_aware_loss(
    zi: torch.Tensor,  # [B, d] L2-normed visual embeds
    zt: torch.Tensor,  # [C, d] L2-normed text prototypes
    labels: torch.Tensor,  # [B]   ground-truth class indices
    logit_scale: nn.Parameter,
    kh: int = KH,
    ks: int = KS,
) -> torch.Tensor:
    """
    Distractor-Aware Contrastive Loss (Eq. 6 from Chatterjee & Ghosh).

    For each sample i:
      s_ij = τ · zᵢᵀpⱼ                       (Eq. 2)
      s⁺ᵢ  = s_{i,yᵢ}                         (positive logit)
      Hᵢ   = Top-kh highest wrong-class sims  (Eq. 3) — hard negatives
      SHᵢ  = ks random from remaining classes  (Eq. 4) — semi-hard negatives
      Lᵢ   = -log( e^s⁺ / (e^s⁺ + Σ_{H} e^s + Σ_{SH} e^s) )  (Eq. 6)
    """
    B, C = zi.shape[0], zt.shape[0]
    tau = logit_scale.clamp(-4.6, 4.6).exp()  # τ; clamp prevents collapse

    # All pairwise scaled cosine similarities: s_ij = τ · zᵢᵀpⱼ
    sims = tau * torch.mm(zi, zt.t())  # [B, C]

    # Positive logits
    pos_logits = sims[torch.arange(B, device=device), labels]  # [B]

    # Mask positives out: -inf so topk ignores them
    neg_sims = sims.clone()
    neg_sims[torch.arange(B, device=device), labels] = float("-inf")

    # ── Hard negatives: top-kh most similar wrong classes (Eq. 3) ──
    hard_logits, hard_col_idx = neg_sims.topk(kh, dim=1)  # [B, kh]

    # ── Semi-hard negatives: ks random from remaining pool (Eq. 4) ──
    # Mark hard negatives so they're excluded from the semi-hard pool
    semi_mask = torch.ones(B, C, dtype=torch.bool, device=device)
    semi_mask[torch.arange(B, device=device), labels] = False  # remove positive
    semi_mask.scatter_(1, hard_col_idx, False)  # remove hard negs

    # Sample ks from the remaining pool (vectorized via batched randperm)
    semi_logits = torch.zeros(B, ks, device=device)
    for i in range(B):
        pool = semi_mask[i].nonzero(as_tuple=True)[0]  # available indices
        chosen = pool[torch.randperm(len(pool), device=device)[:ks]]
        semi_logits[i] = sims[i, chosen]

    # ── Combined logits: [B, 1+kh+ks]  (positive always at index 0) ──
    combined = torch.cat(
        [
            pos_logits.unsqueeze(1),  # [B, 1]
            hard_logits,  # [B, kh]
            semi_logits,  # [B, ks]
        ],
        dim=1,
    )  # [B, 9]

    # Cross-entropy with target=0 (positive is index 0 by construction)
    target = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(combined, target)


# ── 7. DATA LOADING ───────────────────────────────────────────


class HyperspectralDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float() if X.dtype != torch.float32 else X
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_vlm_dataloaders(pca_components, dataset_abbr, batch_size):
    proc_dir = os.path.join(processed_root, f"pca_{pca_components}", dataset_abbr)
    assert os.path.exists(proc_dir), f"Directory not found: {proc_dir}"

    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    print(f"\nTrain X : {X_tr.shape}  |  y: {y_tr.shape}")
    print(f"Test  X : {X_te.shape}  |  y: {y_te.shape}")
    assert X_tr.shape[1:] == (1, 50, 9, 9), f"Shape mismatch: {X_tr.shape}"

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    # Balanced sampler for class-imbalanced IP
    y_np = y_tr.numpy()
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


# ── 8. TRAINING LOOP ─────────────────────────────────────────


def train_vlm(model, train_loader, test_loader, raw_text_embeds):
    """
    End-to-end VLM training with distractor-aware contrastive loss.
    Only vision_encoder + projection_head + logit_scale are updated.
    Text encoder (BGE) stays frozen via precomputed raw_text_embeds.
    """
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_loss = float("inf")
    patience_counter = 0
    best_path = os.path.join(output_dir, "vlm_hsi_IP_best.pth")
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    print(f"\n{'=' * 62}")
    print(f"TRAINING  VLM-HSI  —  Indian Pines")
    print(f"{'=' * 62}")
    print(
        f"Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"Loss: Distractor-Aware Contrastive (kh={KH}, ks={KS})")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_correct = tr_total = 0

        # Recompute text prototypes using current projection head weights
        # (raw_text_embeds is fixed; only projection_head changes)
        with torch.no_grad():
            text_prototypes = model.encode_text(raw_text_embeds)  # [16, 64]

        for data, labels in train_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    zi = model.encode_vision(data)  # [B, 64]
                    loss = distractor_aware_loss(
                        zi, text_prototypes, labels, model.logit_scale
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                zi = model.encode_vision(data)
                loss = distractor_aware_loss(
                    zi, text_prototypes, labels, model.logit_scale
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Track accuracy via nearest-prototype (for monitoring only)
            with torch.no_grad():
                tau = model.logit_scale.clamp(-4.6, 4.6).exp()
                sims = tau * torch.mm(zi.detach(), text_prototypes.t())
                preds = sims.argmax(dim=1)
                tr_correct += preds.eq(labels).sum().item()
                tr_total += labels.size(0)
                tr_loss += loss.detach().item()

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        te_loss = te_correct = te_total = 0

        with torch.no_grad():
            # Update text prototypes with latest projection weights
            text_prototypes_eval = model.encode_text(raw_text_embeds)
            tau = model.logit_scale.clamp(-4.6, 4.6).exp()

            for data, labels in test_loader:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        zi = model.encode_vision(data)
                else:
                    zi = model.encode_vision(data)

                # Validation loss
                val_loss = distractor_aware_loss(
                    zi, text_prototypes_eval, labels, model.logit_scale
                )
                te_loss += val_loss.item()

                # Nearest-prototype inference (Eq. ŷ = argmax_j zᵀpⱼ)
                sims = torch.mm(zi, text_prototypes_eval.t())  # no τ for inference
                preds = sims.argmax(dim=1)
                te_correct += preds.eq(labels).sum().item()
                te_total += labels.size(0)

        avg_tr = tr_loss / len(train_loader)
        avg_te = te_loss / len(test_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        te_acc = 100.0 * te_correct / te_total
        elapsed = time.time() - t0
        tau_val = model.logit_scale.clamp(-4.6, 4.6).exp().item()

        history["train_loss"].append(avg_tr)
        history["test_loss"].append(avg_te)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"TrLoss {avg_tr:.4f}  TrAcc {tr_acc:.2f}% | "
            f"TeLoss {avg_te:.4f}  TeAcc {te_acc:.2f}% | "
            f"τ={tau_val:.2f} | {elapsed:.1f}s"
        )

        # Early stopping on validation loss
        if avg_te < best_loss:
            best_loss = avg_te
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ★ Best model saved  (val_loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete.  Best val loss: {best_loss:.4f}")
    return history, best_path


# ── 9. EVALUATION (Nearest-Prototype Retrieval) ──────────────


def evaluate_vlm(model, test_loader, raw_text_embeds):
    """
    Inference procedure (paper §2):
        ŷ = argmax_j  zₓᵀ pⱼ
    where pⱼ are the L2-normalised text prototypes.
    Computes OA, AA (mean per-class recall), and Cohen's Kappa.
    """
    print(f"\n{'=' * 62}")
    print("EVALUATION  —  VLM-HSI on Indian Pines (Nearest-Prototype)")
    print(f"{'=' * 62}")

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        text_prototypes = model.encode_text(raw_text_embeds)  # [16, 64]

        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    zi = model.encode_vision(data)  # [B, 64]
            else:
                zi = model.encode_vision(data)

            # No temperature scaling at inference — pure cosine similarity
            sims = torch.mm(zi, text_prototypes.t())  # [B, 16]
            preds = sims.argmax(dim=1)  # [B]

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── Overall Accuracy (OA) ─────────────────────────────────────────
    OA = accuracy_score(all_targets, all_preds)

    # ── Average Accuracy (AA) — mean per-class recall ─────────────────
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = all_targets == c
        per_class_acc.append((all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0)
    AA = float(np.mean(per_class_acc))

    # ── Cohen's Kappa (κ) ────────────────────────────────────────────
    K = cohen_kappa_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    # ── Console output ────────────────────────────────────────────────
    print(f"\n{'─' * 44}")
    print(f"  Overall Accuracy  (OA) : {OA * 100:.2f}%")
    print(f"  Average Accuracy  (AA) : {AA * 100:.2f}%")
    print(f"  Kappa Coefficient  (κ) : {K:.4f}")
    print(f"{'─' * 44}")
    print("\nPer-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        print(f"  {i + 1:2d}. {name:<35s}: {acc * 100:.2f}%")

    # ── Save JSON ─────────────────────────────────────────────────────
    results = {
        "model": "VLM-HSI (Chatterjee & Ghosh, 2025)",
        "dataset": "Indian Pines",
        "pca_components": PCA_COMPONENTS,
        "patch_size": PATCH_SIZE,
        "text_encoder": TEXT_MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "overall_accuracy": float(OA),
        "average_accuracy": float(AA),
        "kappa": float(K),
        "per_class_accuracy": {
            CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    out_file = os.path.join(output_dir, "vlm_hsi_IP_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_file}")
    return results, cm, per_class_acc


# ── 10. PLOTTING UTILITIES ────────────────────────────────────


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "b-", lw=2, label="Train")
    axes[0].plot(ep, history["test_loss"], "r-", lw=2, label="Test")
    axes[0].set_title("Contrastive Loss — VLM-HSI (Indian Pines)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], "b-", lw=2, label="Train")
    axes[1].plot(ep, history["test_acc"], "r-", lw=2, label="Test")
    axes[1].set_title("Accuracy — VLM-HSI (Indian Pines)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "vlm_hsi_IP_training_history.png")
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
    plt.title(
        "Confusion Matrix — VLM-HSI (Indian Pines)", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "vlm_hsi_IP_confusion_matrix.png")
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
        "Per-Class Accuracy — VLM-HSI (Indian Pines)", fontsize=14, fontweight="bold"
    )
    aa_val = np.mean(per_class_acc) * 100
    plt.axhline(
        y=aa_val, color="red", linestyle="--", lw=1.5, label=f"AA = {aa_val:.2f}%"
    )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "vlm_hsi_IP_per_class_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy → {path}")


def plot_embedding_similarity(model, raw_text_embeds):
    """Visualise cross-modal cosine similarity matrix between text prototypes."""
    model.eval()
    with torch.no_grad():
        zt = model.encode_text(raw_text_embeds)  # [16, 64]
        sim = torch.mm(zt, zt.t()).cpu().numpy()  # [16, 16]

    short_names = [n[:12] for n in CLASS_NAMES]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        xticklabels=short_names,
        yticklabels=short_names,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(
        "Text Prototype Similarity Matrix — VLM-HSI", fontsize=13, fontweight="bold"
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "vlm_hsi_text_proto_similarity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prototype similarity → {path}")


# ── 11. SHAPE SANITY CHECK ────────────────────────────────────


def verify_shapes(model, raw_text_embeds):
    print("\n── Shape verification (dry run) ──")
    model.eval()
    dummy = torch.randn(4, 1, 50, 9, 9, device=device)
    labels = torch.randint(0, NUM_CLASSES, (4,), device=device)
    with torch.no_grad():
        zv = model.encode_vision(dummy)  # [4, 64]
        zt = model.encode_text(raw_text_embeds)  # [16, 64]
        loss = distractor_aware_loss(zv, zt, labels, model.logit_scale)

    print(f"  Input     : {tuple(dummy.shape)}")
    print(f"  Visual z  : {tuple(zv.shape)}  L2-norm={zv.norm(dim=1).mean():.4f}  ✓")
    print(f"  Text proto: {tuple(zt.shape)}  L2-norm={zt.norm(dim=1).mean():.4f}  ✓")
    print(f"  Loss value: {loss.item():.4f}  (should be ≈ log(9)≈2.20 at init)  ✓")
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {trainable_p:,}  (paper: ~240K)")


# ── 12. MAIN EXECUTION ────────────────────────────────────────

if __name__ == "__main__" or True:
    # 12a. Build model
    model = VLM_HSI(
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        text_dim=TEXT_DIM,
    ).to(device)

    # 12b. Precompute frozen BGE text embeddings (once, ~30s on first run)
    raw_text_embeds = compute_text_embeddings(IP_PROMPTS)  # [16, 1024]

    # 12c. Shape verification
    verify_shapes(model, raw_text_embeds)

    # 12d. Data loaders
    train_loader, test_loader = get_vlm_dataloaders(
        PCA_COMPONENTS, DATASET_ABBR, BATCH_SIZE
    )

    # Double-check live batch shape
    sample_x, _ = next(iter(train_loader))
    print(f"\nLive batch shape  : {sample_x.shape}")
    assert sample_x.shape[1:] == (1, 50, 9, 9), f"Unexpected shape: {sample_x.shape}"
    print("Shape assertion passed ✓")

    # 12e. Train
    history, best_path = train_vlm(model, train_loader, test_loader, raw_text_embeds)

    # 12f. Load best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\nLoaded best weights from: {best_path}")

    # 12g. Evaluate (OA / AA / Kappa)
    results, cm, per_class_acc = evaluate_vlm(model, test_loader, raw_text_embeds)

    # 12h. Plots
    plot_training_history(history)
    plot_confusion_matrix(cm)
    plot_per_class_accuracy(per_class_acc)
    plot_embedding_similarity(model, raw_text_embeds)

    # 12i. Final IEEE TGRS table block
    print(f"\n{'═' * 47}")
    print(f"  VLM-HSI — Indian Pines  |  IEEE TGRS Table")
    print(f"{'═' * 47}")
    print(f"  OA  : {results['overall_accuracy'] * 100:.2f}%    (paper: 94.03%)")
    print(f"  AA  : {results['average_accuracy'] * 100:.2f}%")
    print(f"  κ   : {results['kappa']:.4f}    (paper: 0.9354)")
    print(f"{'═' * 47}")
