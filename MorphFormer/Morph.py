import os, sys
import argparse
import time
import gzip
import shutil
import json
import logging
from datetime import datetime
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, Softmax
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

# ==============================================================================
# 1. SETUP AND UTILS INTEGRATION
# ==============================================================================
base_input = "/kaggle/input/datasets/adityachaudhary1306/utility-scripts"
dest_root = "/kaggle/working"

folders_to_move = ["utils"]
for folder in folders_to_move:
    src = os.path.join(base_input, folder)
    dst = os.path.join(dest_root, folder)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst)
        print(f"✅ Moved {folder} to {dst}")
    except FileNotFoundError:
        print(
            f"⚠️ Utility scripts not found at {src}. Ensure the path is correct or MetricsTracker is defined."
        )

try:
    from utils.metrics_utils import MetricsTracker
except ImportError:
    print(
        "⚠️ Could not import MetricsTracker. Please ensure it is in the correct directory."
    )

# Memory optimization settings
MEMORY_OPTIMIZATION = {
    "gradient_accumulation_steps": 1,
    "enable_mixed_precision": True,
    "enable_gradient_checkpointing": True,
    "clear_cache_frequency": 10,
    "persistent_workers": True,
    "prefetch_factor": 2,
}

DATASET_CONFIGS = {
    "IP": {
        "name": "Indian Pines",
        "num_classes": 16,
        "class_names": [
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
        ],
        "spectral_dim": 200,
        "batch_size": 16,
    }
}

processed_root = "/kaggle/input/datasets/adityachaudhary1306/hsi-ds-pca-50-train-5/"
model_dir = "/kaggle/working/best_models"
results_dir = "/kaggle/working/results"
logs_dir = "/kaggle/working/logs"

for directory in [model_dir, results_dir, logs_dir]:
    os.makedirs(directory, exist_ok=True)
    if directory == results_dir:
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

logger = train_logger = eval_logger = data_logger = model_logger = None

# ==============================================================================
# 2. MORPHFORMER ARCHITECTURE (MATHEMATICALLY CORRECTED)
# ==============================================================================
FM = 16


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))


class Morphology(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        soft_max=True,
        beta=15,
        type=None,
    ):
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        x = fixed_padding(x, self.kernel_size, dilation=1)
        x = self.unfold(x)
        x = x.unsqueeze(1)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))
        weight = self.weight.view(self.out_channels, -1)
        weight = weight.unsqueeze(0).unsqueeze(-1)
        if self.type == "erosion2d":
            x = weight - x
        elif self.type == "dilation2d":
            x = weight + x
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta

        if self.type == "erosion2d":
            x = -1 * x
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)
        return x


class Dilation2d(Morphology):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20
    ):
        super(Dilation2d, self).__init__(
            in_channels, out_channels, kernel_size, soft_max, beta, "dilation2d"
        )


class Erosion2d(Morphology):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20
    ):
        super(Erosion2d, self).__init__(
            in_channels, out_channels, kernel_size, soft_max, beta, "erosion2d"
        )


class HetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=None,
        p=64,
        g=64,
    ):
        super(HetConv, self).__init__()
        self.gwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=g,
            padding=kernel_size // 3,
            stride=stride,
        )
        self.pwc = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, groups=p, stride=stride
        )

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class SpectralMorph(nn.Module):
    def __init__(self, FM, NC, kernel=3):
        super(SpectralMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM, FM, 1, padding=0)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM, FM, 1, padding=0)

    def forward(self, x):
        z1 = self.conv1(self.erosion(x))
        z2 = self.conv2(self.dilation(x))
        return z1 + z2


class SpatialMorph(nn.Module):
    def __init__(self, FM, NC, kernel=3):
        super(SpatialMorph, self).__init__()
        self.erosion = Erosion2d(NC, FM, kernel, soft_max=False)
        self.conv1 = nn.Conv2d(FM, FM, 3, padding=1)
        self.dilation = Dilation2d(NC, FM, kernel, soft_max=False)
        self.conv2 = nn.Conv2d(FM, FM, 3, padding=1)

    def forward(self, x):
        z1 = self.conv1(self.erosion(x))
        z2 = self.conv2(self.dilation(x))
        return z1 + z2


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.wk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.wv(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(self, dim, blockNum=0):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        kernels = [3, 5]
        self.cls_norm = LayerNorm(dim, eps=1e-6)
        self.spec_morph = nn.Sequential(
            SpectralMorph(FM, FM * 2, kernels[blockNum]), nn.BatchNorm2d(FM), nn.GELU()
        )
        self.spat_morph = nn.Sequential(
            SpatialMorph(FM, FM * 2, kernels[blockNum]), nn.BatchNorm2d(FM), nn.GELU()
        )
        self.attn = CrossAttention(dim)

    def forward(self, x):
        ht, w = x.shape[2:]
        rest = x[:, 1:]
        rest1 = self.spec_morph(rest)
        rest2 = self.spat_morph(rest)
        rest = torch.cat([rest1, rest2], dim=1)
        x = torch.cat([x[:, 0:1, :], rest], dim=1)
        clsTok = x[:, 0:1]
        h = clsTok
        clsTok = self.attn(
            self.attention_norm(x.reshape(x.shape[0], x.shape[1], -1))
        ).reshape(x.shape[0], 1, ht, w)
        clsTok = clsTok + h
        clsTok = self.cls_norm(
            clsTok.reshape(clsTok.shape[0], clsTok.shape[1], -1)
        ).reshape(clsTok.shape)
        x = torch.cat([clsTok, x[:, 1:]], dim=1)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.ModuleList([Block(dim, i) for i in range(2)])
        self.encoder_norm = LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.encoder_norm(x)
        return x[:, 0]


class MorphFormer(nn.Module):
    def __init__(self, FM, NC, Classes):
        super(MorphFormer, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            HetConv(
                8 * (NC - 8),
                FM * 4,
                p=1,
                g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
            ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
        )
        self.ca = CrossAttentionBlock(FM * 4)

        self.out3 = nn.Linear(FM * 4, Classes)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, FM * 4))
        self.position_embeddings = nn.Parameter(torch.zeros(1, FM * 2 + 1, FM * 4))
        self.dropout = nn.Dropout(0.1)
        self.FM = FM
        self.token_wA = nn.Parameter(torch.empty(1, FM * 2, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x1):
        # 1. Adapt to varying spatial sizes dynamically instead of hardcoding 11x11
        if x1.dim() == 4:
            x1 = x1.unsqueeze(1)  # [B, 1, NC, H, W]

        spatial_h, spatial_w = x1.shape[-2], x1.shape[-1]

        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, spatial_h, spatial_w)
        x1 = self.conv6(x1)

        cls_tokens = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)

        wa = self.token_wA.expand(x1.shape[0], -1, -1)
        wa = rearrange(wa, "b h w -> b w h")
        A = torch.einsum("bij,bjk->bik", x1, wa)
        A = rearrange(A, "b h w -> b w h")
        A = A.softmax(dim=-1)

        wv = self.token_wV.expand(x1.shape[0], -1, -1)
        VV = torch.einsum("bij,bjk->bik", x1, wv)
        T = torch.einsum("bij,bjk->bik", A, VV)

        x = torch.cat((cls_tokens, T), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        x = embeddings.reshape(
            embeddings.shape[0],
            embeddings.shape[1],
            int(math.sqrt(self.FM * 4)),
            int(math.sqrt(self.FM * 4)),
        )
        x = self.ca(x)
        x = x.reshape(x.shape[0], -1)

        # 2. Bug Fix: Route the CLS token through the actual classifier head
        x = self.out3(x)
        return x


class MorphFormerWrapper(nn.Module):
    """Wraps MorphFormer to output (loss, logits, components) expected by the training loop"""

    def __init__(self, FM, NC, Classes):
        super().__init__()
        self.model = MorphFormer(FM, NC, Classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        logits = self.model(x)
        if target is not None:
            loss = self.criterion(logits, target)
            loss_components = {
                "loss_cls": loss,
                "loss_rec": torch.tensor(0.0, device=loss.device),
                "loss_con": torch.tensor(0.0, device=loss.device),
            }
            return loss, logits, loss_components
        return None, logits, None


# ==============================================================================
# 3. TRAINING LOOP IMPLEMENTATION
# ==============================================================================
def get_dataloaders(pca_components, dataset_abbr, batch_size=16):
    proc_dir = os.path.join(processed_root, f"pca_{pca_components}", dataset_abbr)
    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    train_ds = torch.utils.data.TensorDataset(X_tr.float(), y_tr.long())
    test_ds = torch.utils.data.TensorDataset(X_te.float(), y_te.long())

    class_counts = np.bincount(
        y_tr.numpy(), minlength=DATASET_CONFIGS[dataset_abbr]["num_classes"]
    )
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    valid_mask = class_counts > 0
    class_weights[valid_mask] = 1.0 / class_counts[valid_mask]

    sample_weights = torch.from_numpy(class_weights[y_tr.numpy()]).float()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


def evaluate_model(model, test_loader, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for data, target in test_loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            _, logits, _ = model(data)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)

    overall_acc = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)

    per_class_acc = []
    for i in range(num_classes):
        if np.sum(all_targets == i) > 0:
            per_class_acc.append(
                np.sum((all_targets == i) & (all_preds == i)) / np.sum(all_targets == i)
            )
        else:
            per_class_acc.append(0.0)

    avg_per_class_acc = np.mean(per_class_acc)
    return overall_acc, avg_per_class_acc, kappa, per_class_acc


def train_morphformer():
    dataset_abbr = "IP"
    config = DATASET_CONFIGS[dataset_abbr]
    pca_comps = 50
    epochs = 100
    batch_size = config["batch_size"]

    print(f"Loading {config['name']} Dataset...")
    train_loader, test_loader = get_dataloaders(pca_comps, dataset_abbr, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating morphFormer Model on {device}...")
    model = MorphFormerWrapper(
        FM=16, NC=config["spectral_dim"], Classes=config["num_classes"]
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_test_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    print(f"{'=' * 60}\nSTARTING TRAINING: morphFormer on {config['name']}\n{'=' * 60}")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    loss, logits, _ = model(data, target)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, logits, _ = model(data, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            train_correct += logits.argmax(dim=1).eq(target).sum().item()
            train_total += target.size(0)

        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = (
                    data.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                )
                loss, logits, _ = model(data, target)
                test_loss += loss.item()
                test_correct += logits.argmax(dim=1).eq(target).sum().item()
                test_total += target.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()

    print(f"\n{'-' * 60}\nEvaluating Best Model...\n{'-' * 60}")
    model.load_state_dict(best_model_state)
    oa, aa, kappa, per_class_acc = evaluate_model(model, test_loader, config["num_classes"])

    print(f"🎯 Final Results on {config['name']}:")
    print(f"  ➜ Overall Accuracy (OA): {oa * 100:.2f}%")
    print(f"  ➜ Average Accuracy (AA): {aa * 100:.2f}%")
    print(f"  ➜ Cohen's Kappa (K):     {kappa:.4f}\n")
    print("  ➜ Per-Class Accuracies:")
    for i, c_acc in enumerate(per_class_acc):
        class_name = config['class_names'][i] if 'class_names' in config and i < len(config['class_names']) else str(i)
        print(f"      Class {class_name}: {c_acc * 100:.2f}%")


if __name__ == "__main__":
    train_morphformer()
