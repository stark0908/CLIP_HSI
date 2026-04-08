import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from transformers import CLIPTokenizer

# ==============================================================================
# 1. CONFIGURATION & MEMORY OPTIMIZATION
# ==============================================================================
MEMORY_OPTIMIZATION = {
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
        "spectral_dim": 200,  # Original dim, but we will use 50 PCA components
        "batch_size": 32,
    }
}

PROCESSED_ROOT = "/home/23dcs505/datasets/IP"


# ==============================================================================
# 2. DATA LOADING PIPELINE
# ==============================================================================
class HyperspectralDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

        # Memory optimization
        if self.X.dtype == torch.float64:
            self.X = self.X.float()
        if self.y.dtype == torch.int64 and self.y.max() < 256:
            self.y = self.y.byte()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx].long() if self.y.dtype == torch.uint8 else self.y[idx]
        return x, y


def get_dataloaders(pca_components, dataset_abbr, batch_size=32):
    print(f"Loading dataset: {dataset_abbr} with {pca_components} PCA components")
    proc_dir = os.path.join(PROCESSED_ROOT, f"pca_{pca_components}", dataset_abbr)

    if not os.path.exists(proc_dir):
        raise FileNotFoundError(f"Dataset directory {proc_dir} not found! Check paths.")

    # Load tensors
    load_start = time.time()
    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))
    print(f"Data loaded successfully in {time.time() - load_start:.2f}s")
    print(f"Data shapes - Train: {X_tr.shape}, Test: {X_te.shape}")

    dataset_config = DATASET_CONFIGS.get(dataset_abbr, {})

    train_ds = HyperspectralDataset(X_tr, y_tr)
    test_ds = HyperspectralDataset(X_te, y_te)

    # WeightedRandomSampler for class imbalance
    y_train_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else y_tr
    class_counts = np.bincount(
        y_train_np, minlength=dataset_config.get("num_classes", 0)
    )
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    valid_mask = class_counts > 0
    class_weights[valid_mask] = 1.0 / class_counts[valid_mask]

    sample_weights_np = class_weights[y_train_np]
    samples_weights = torch.from_numpy(sample_weights_np).float()

    sampler = WeightedRandomSampler(
        weights=samples_weights, num_samples=len(samples_weights), replacement=True
    )
    print("  ⚖️ WeightedRandomSampler initialized for balanced training")

    cpu_count = os.cpu_count()
    num_workers = min(4, cpu_count // 2) if cpu_count and cpu_count > 2 else 2
    pin_memory = torch.cuda.is_available()
    persistent_workers = MEMORY_OPTIMIZATION["persistent_workers"] and num_workers > 0
    prefetch_factor = (
        MEMORY_OPTIMIZATION["prefetch_factor"] if num_workers > 0 else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    print(
        f"[LOADERS] Train:{len(train_loader)} batches | Test:{len(test_loader)} batches"
    )
    return train_loader, test_loader


# ==============================================================================
# 3. MATHEMATICALLY CORRECTED 3D-CNN VISUAL ENCODER
# ==============================================================================
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm3d(out_channel),
    )
    return layer


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()
        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)
        out = F.relu(x1 + x3, inplace=True)
        return out


class D_Res_3d_CNN(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel1,
        out_channel2,
        CLASS_NUM,
        patch_size,
        n_bands,
        embed_dim,
    ):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands
        self.patch_size = patch_size

        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2)
        )
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.conv1 = nn.Conv3d(
            in_channels=out_channel2, out_channels=32, kernel_size=(1, 3, 3), bias=False
        )

        self.flat_size = self._get_layer_size()

        self.fc = nn.Linear(
            in_features=self.flat_size, out_features=embed_dim, bias=False
        )
        self.classifier = nn.Linear(
            in_features=self.flat_size, out_features=CLASS_NUM, bias=False
        )

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.n_bands, self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0], -1)
            s = x.size()[1]
        return s

    def forward(self, x):
        # Input: [B, 1, 50, 9, 9] (Unsqueeze removed to fix dimension mismatch)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        y = self.classifier(x)
        proj = self.fc(x)
        return y, proj


# ==============================================================================
# 4. TRANSFORMER TEXT ENCODER & LDGNET ARCHITECTURE
# ==============================================================================
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class LDGnet(nn.Module):
    def __init__(
        self,
        embed_dim,
        inchannel,
        vision_patch_size,
        num_classes,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    ):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.visual = D_Res_3d_CNN(
            1, 8, 16, num_classes, vision_patch_size, inchannel, embed_dim
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(
        self,
        image,
        text_prototypes,
        label,
        text_q1_prototypes=None,
        text_q2_prototypes=None,
    ):
        image_prob, image_features = self.encode_image(image)
        if self.training:
            text_features = self.encode_text(text_prototypes)
            text_features_q1 = self.encode_text(text_q1_prototypes)
            text_features_q2 = self.encode_text(text_q2_prototypes)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features_q1 = text_features_q1 / text_features_q1.norm(
                dim=1, keepdim=True
            )
            text_features_q2 = text_features_q2 / text_features_q2.norm(
                dim=1, keepdim=True
            )

            logit_scale = self.logit_scale.exp()

            logits_img = logit_scale * image_features @ text_features.t()
            loss_clip = F.cross_entropy(logits_img, label.long())

            logits_q1 = logit_scale * image_features @ text_features_q1.t()
            loss_q1 = F.cross_entropy(logits_q1, label.long())

            logits_q2 = logit_scale * image_features @ text_features_q2.t()
            loss_q2 = F.cross_entropy(logits_q2, label.long())

            return loss_clip, (loss_q1 + loss_q2) / 2, image_prob
        else:
            return torch.tensor(0.0).to(image.device), image_prob


# ==============================================================================
# 5. TEXT PROMPTS & TOKENIZATION
# ==============================================================================
INDIAN_PINES_PROMPTS = {
    "coarse": [
        "This is a field of alfalfa.",
        "This is a corn field with no-till farming.",
        "This is a corn field with minimum tillage.",
        "This is a conventional corn field.",
        "This is a grass pasture.",
        "This is a grassy area with trees.",
        "This is a mowed grass pasture.",
        "This is a field of windrowed hay.",
        "This is a field of oats.",
        "This is a soybean field with no-till farming.",
        "This is a soybean field with minimum tillage.",
        "This is a cleanly tilled soybean field.",
        "This is a wheat field.",
        "This is a wooded area.",
        "This is a mixed developed area.",
        "These are stone or steel towers.",
    ],
    "fine1": [
        "Lush green alfalfa plants growing densely in an agricultural field.",
        "A field of corn grown using no-till agricultural practices, leaving crop residue.",
        "Corn plants growing in a field with minimal soil disturbance.",
        "A standard agricultural field densely planted with mature corn.",
        "An open area of land covered with natural pasture grass for grazing.",
        "A mixed landscape featuring both pasture grass and scattered trees.",
        "A field of pasture grass that has been recently cut and maintained.",
        "Cut hay gathered into long rows to dry in the field.",
        "An agricultural area covered with growing oat crops.",
        "Soybean plants emerging from unplowed soil with previous crop residue.",
        "Soybeans grown in agricultural soil with reduced tillage practices.",
        "A conventionally plowed field cleanly planted with rows of soybeans.",
        "A broad agricultural field dedicated to the cultivation of wheat.",
        "A dense concentration of trees forming a small forest or woodland.",
        "A suburban landscape containing buildings, driveways, trees, and grass.",
        "Human-constructed towers made of stone or steel materials.",
    ],
    "fine2": [
        "A forage crop field primarily covered with alfalfa vegetation.",
        "Unplowed soil supporting the growth of a corn crop.",
        "Agricultural land showing corn crops cultivated with conservation tillage.",
        "Rows of tall corn stalks growing in a plowed agricultural field.",
        "A green field dominated by grazing grasses and natural vegetation.",
        "Vegetation consisting of a grassy understory with a canopy of trees.",
        "Short, trimmed grass in an agricultural grazing pasture.",
        "Agricultural field showing parallel lines of harvested and windrowed hay.",
        "A cereal grain field characterized by the cultivation of oats.",
        "A field of soybeans cultivated using environmentally friendly no-till methods.",
        "A conservation tillage field actively growing soybean crops.",
        "Agricultural land with bare soil between rows of growing soybean plants.",
        "Dense growth of wheat crops covering a farmland area.",
        "Natural land cover dominated by thick tree canopy and woody vegetation.",
        "Human-made structures intermixed with natural grassy and wooded elements.",
        "Industrial or architectural structures built from solid steel and stone.",
    ],
}


def get_text_tokens(sentences, max_length=77):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = tokenizer(
        sentences,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return tokens["input_ids"]


# ==============================================================================
# 6. TRAINING & EVALUATION LOOP
# ==============================================================================
def train_ldgnet(model, train_loader, test_loader, device, epochs=100, lr=1e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-tokenize class prototypes and send to device
    print("Tokenizing text prompts...")
    text_coarse = get_text_tokens(INDIAN_PINES_PROMPTS["coarse"]).to(device)
    text_f1 = get_text_tokens(INDIAN_PINES_PROMPTS["fine1"]).to(device)
    text_f2 = get_text_tokens(INDIAN_PINES_PROMPTS["fine2"]).to(device)

    best_oa = 0.0

    print("🚀 Starting LDGNet Training...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            loss_clip, loss_q, image_prob = model(
                data, text_coarse, target, text_f1, text_f2
            )
            loss_cls = F.cross_entropy(image_prob, target.long())
            total_loss = loss_cls + loss_clip + loss_q
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pred = image_prob.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        train_acc = 100.0 * train_correct / train_total
        oa, aa, kappa = evaluate_ldgnet(
            model, test_loader, device, num_classes=16, silent=True
        )

        if oa > best_oa:
            best_oa = oa
            # torch.save(model.state_dict(), 'ldgnet_best_ip.pth')

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Time: {time.time() - start_time:.1f}s | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val OA: {oa * 100:.2f}% (Best: {best_oa * 100:.2f}%)"
        )

    print("\n🏁 Final Model Evaluation:")
    evaluate_ldgnet(model, test_loader, device, num_classes=16, silent=False)


def evaluate_ldgnet(model, test_loader, device, num_classes=16, silent=True):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, image_prob = model(data, None, None)
            pred = image_prob.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    oa = accuracy_score(all_targets, all_preds)

    per_class_acc = []
    for i in range(num_classes):
        if np.sum(all_targets == i) > 0:
            acc = np.sum((all_targets == i) & (all_preds == i)) / np.sum(
                all_targets == i
            )
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(all_targets, all_preds)

    if not silent:
        print("-" * 40)
        print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
        print(f"Average Accuracy (AA): {aa * 100:.2f}%")
        print(f"Cohen's Kappa (K)  :   {kappa:.4f}")
        print("-" * 40)

    return oa, aa, kappa


# ==============================================================================
# 7. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Get DataLoaders (PCA=50, Indian Pines)
    train_loader, test_loader = get_dataloaders(
        pca_components=50, dataset_abbr="IP", batch_size=32
    )

    # 3. Initialize Model
    model = LDGnet(
        embed_dim=512,
        inchannel=50,  # 50 PCA Spectral Bands
        vision_patch_size=9,  # 9x9 Spatial Patches
        num_classes=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )

    # 4. Train Model
    train_ldgnet(model, train_loader, test_loader, device, epochs=100)
