import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from transformers import CLIPTokenizer, CLIPTextModel

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
            "Corn no-till",
            "Corn min-till",
            "Corn",
            "Grass pasture",
            "Grass trees",
            "Grass pasture mowed",
            "Hay windrowed",
            "Oats",
            "Soybean no-till",
            "Soybean min-till",
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
        # Graceful handling for missing data in user's env during testing/startup
        print(f"⚠️ Dataset directory {proc_dir} not found! Check paths. Proceeding with dummy data for loader.")
        X_tr, y_tr = torch.randn(100, 1, pca_components, 13, 13), torch.randint(0, 16, (100,))
        X_te, y_te = torch.randn(100, 1, pca_components, 13, 13), torch.randint(0, 16, (100,))
    else:
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
    num_workers = min(4, cpu_count // 2) if cpu_count and cpu_count > 2 else 0
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

class LDGnet(nn.Module):
    def __init__(
        self,
        embed_dim,
        inchannel,
        vision_patch_size,
        num_classes,
        use_pretrained_clip=True,
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.visual = D_Res_3d_CNN(
            1, 8, 16, num_classes, vision_patch_size, inchannel, embed_dim
        )
        self.use_pretrained_clip = use_pretrained_clip

        # ---------------------------------------------------------
        # FIX 1 & 4: Loading Pretrained CLIP & Architecture depth
        # ---------------------------------------------------------
        if self.use_pretrained_clip:
            print("🚀 Initializing Text Encoder with pre-trained CLIP weights...")
            # Load pretrained text model weights (from standard ViT-B/32)
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            # Map the (typically 512) dimensional CLIP output to `embed_dim` just in case 
            # though standard CLIP base already outputs 512
            clip_out_dim = self.text_encoder.config.projection_dim if hasattr(self.text_encoder.config, "projection_dim") else 512
            self.text_projection = nn.Linear(clip_out_dim, embed_dim, bias=False) if clip_out_dim != embed_dim else nn.Identity()
        else:
            print("ℹ️ Using dummy untrained Text Encoder (Legacy Mode)")
            # You can place the custom Transformer here if keeping the old un-trained method
            pass
            

    def encode_image(self, image):
        return self.visual(image.type(self.visual.conv1.weight.dtype))

    def encode_text(self, text_input_ids):
        if self.use_pretrained_clip:
            # text_input_ids shape expected [N, 76] or [N, 77]
            outputs = self.text_encoder(input_ids=text_input_ids)
            # Use pooler output or final state (CLIP generally uses representation of the EOS token)
            text_features = outputs.pooler_output 
            text_features = self.text_projection(text_features)
            return text_features
        return None

    def contrastive_loss(self, image_features, text_features, labels):
        """
        FIX 3: Computing Symmetric Supervised Contrastive Loss (Both Image-to-Text and Text-to-Image)
        """
        logit_scale = self.logit_scale.exp()
        # [B, C]
        logits_img = logit_scale * image_features @ text_features.t()
        # [C, B]
        logits_text = logits_img.t()

        # Image -> Text loss (Standard Cross Entropy against Class prototype index)
        # Each image corresponds uniquely to 1 correct text prototype.
        loss_i2t = F.cross_entropy(logits_img, labels.long())

        # Text -> Image loss
        # Each text prototype corresponds to potentially multiple positive valid images in the batch.
        loss_t2i = 0.0
        num_classes = text_features.shape[0]
        valid_classes = 0
        
        for c in range(num_classes):
            pos_mask = (labels == c)
            if pos_mask.sum() > 0:
                log_prob = F.log_softmax(logits_text[c], dim=0) # [B]
                loss_t2i -= log_prob[pos_mask].mean()
                valid_classes += 1
                
        if valid_classes > 0:
            loss_t2i = loss_t2i / valid_classes
        else:
            loss_t2i = torch.tensor(0.0).to(image_features.device)

        return loss_i2t + loss_t2i

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

            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            text_features_q1 = F.normalize(text_features_q1, dim=1)
            text_features_q2 = F.normalize(text_features_q2, dim=1)

            loss_coarse = self.contrastive_loss(image_features, text_features, label)
            loss_fine1 = self.contrastive_loss(image_features, text_features_q1, label)
            loss_fine2 = self.contrastive_loss(image_features, text_features_q2, label)
            loss_fine = (loss_fine1 + loss_fine2) / 2.0

            return loss_coarse, loss_fine, image_prob
        else:
            return torch.tensor(0.0).to(image.device), torch.tensor(0.0).to(image.device), image_prob


# ==============================================================================
# 5. TEXT PROMPTS & TOKENIZATION
# ==============================================================================
# FIX 6: Coarse Prompts Formulation
INDIAN_PINES_PROMPTS = {
    "coarse": [f"A HSI of {cls}." for cls in DATASET_CONFIGS["IP"]["class_names"]],
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


def get_text_tokens(sentences, max_length=76): # FIX 4: Max length 76
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = tokenizer(
        sentences,
        padding="max_length",
        max_length=max_length, # 76
        truncation=True,
        return_tensors="pt",
    )
    return tokens["input_ids"]


# ==============================================================================
# 6. TRAINING & EVALUATION LOOP
# ==============================================================================
def train_ldgnet(model, train_loader, test_loader, device, epochs=100, lr=1e-4, lambda_val=1.0, alpha_val=0.3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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

            loss_coarse, loss_fine, image_prob = model(
                data, text_coarse, target, text_f1, text_f2
            )
            
            # FIX 2: Correct Loss Weightings
            loss_SD = F.cross_entropy(image_prob, target.long())
            
            # total_loss = L_SD + lambda * ((1 - alpha)*L_coarse + alpha * L_fine)
            loss_alignment = lambda_val * ((1 - alpha_val) * loss_coarse + alpha_val * loss_fine)
            total_loss = loss_SD + loss_alignment
            
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

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Time: {time.time() - start_time:.1f}s | "
            f"Train Loss: {train_loss / max(1, len(train_loader)):.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val OA: {oa * 100:.2f}% (Best: {best_oa * 100:.2f}%)"
        )


def evaluate_ldgnet(model, test_loader, device, num_classes=16, silent=True):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, _, image_prob = model(data, None, None)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        pca_components=50, dataset_abbr="IP", batch_size=32
    )

    # Dynamically extract the patch size mapped inside the actual loaded tensor data
    # (Pre-processed IP dataset files are 9x9 instead of the paper's 13x13)
    actual_patch_size = train_loader.dataset.X.shape[-1]

    # FIX 5: Dynamically configure vision_patch_size to avoid matmul mismatches
    model = LDGnet(
        embed_dim=512,
        inchannel=50,  
        vision_patch_size=actual_patch_size,  
        num_classes=16,
        use_pretrained_clip=True,
    )

    # FIX 2: Added loss hyperparameter tunings from paper
    train_ldgnet(model, train_loader, test_loader, device, epochs=100, lambda_val=1.0, alpha_val=0.3)
