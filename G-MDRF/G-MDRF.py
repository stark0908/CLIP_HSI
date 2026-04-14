# !pip install gpytorch transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
from transformers import Adafactor
import numpy as np
import time
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ==========================================
# 1. Custom Composite Covariance Function
# ==========================================
class GaussianMaternKernel(Kernel):
    """
    Composite kernel combining Matern (h=5/2) and RBF (Gaussian)
    as described in the G-MDRF paper.
    """

    has_lengthscale = True

    def __init__(self, ard_num_dims=None, **kwargs):
        super(GaussianMaternKernel, self).__init__(**kwargs)
        # The paper specifies h = 5/2 for the Matern component
        self.matern = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, **kwargs)
        self.rbf = RBFKernel(ard_num_dims=ard_num_dims, **kwargs)

        # Fluctuation parameter xi (ξ) from the paper
        self.register_parameter(
            name="raw_xi", parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

    @property
    def xi(self):
        return torch.nn.functional.softplus(self.raw_xi)

    def forward(self, x1, x2, diag=False, **params):
        # k(x, x') = Matern(x, x') * (xi^2 * RBF(x, x'))
        matern_covar = self.matern(x1, x2, diag=diag, **params)
        rbf_covar = self.rbf(x1, x2, diag=diag, **params)

        if diag:
            return matern_covar * self.xi * rbf_covar
        else:
            return matern_covar.mul(rbf_covar) * self.xi


# ==========================================
# 2. Spectral Field Modeling (SIVGP)
# ==========================================
class SIVGPModel(ApproximateGP):
    def __init__(self, inducing_points, num_classes):
        # Construct the variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_classes])
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_tasks=num_classes,
            )
        )
        super(SIVGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_classes])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GaussianMaternKernel(ard_num_dims=50, batch_shape=torch.Size([num_classes])),
            batch_shape=torch.Size([num_classes]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 3. Spatial Field Modeling (SAMRF via ADMM)
# ==========================================
def samrf_admm_patch(f_gp, delta_sp=10.0, phi=0.1, max_iter=10):
    """
    Localized ADMM optimization for the SAMRF spatial prior over a 9x9 patch.
    f_gp: Spectral latent function predictions [batch_size, num_classes, h, w]
    """
    batch_size, num_classes, h, w = f_gp.shape
    device = f_gp.device

    # Initialize variables per Algorithm 1
    t = f_gp.clone()  # t^(0) = f_GP
    Q1 = f_gp.clone()
    Q2_h = torch.zeros(batch_size, num_classes, h, w).to(device)
    Q2_v = torch.zeros(batch_size, num_classes, h, w).to(device)
    Q3 = f_gp.clone()
    
    H1 = torch.zeros_like(t)
    H2_h = torch.zeros_like(t)
    H2_v = torch.zeros_like(t)
    H3 = torch.zeros_like(t)
    
    gamma = 1.0
    threshold = delta_sp / phi
    
    import torch.nn.functional as F

    def apply_F(x):
        # Forward finite differences
        dh = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dv = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return dh, dv

    def apply_FT(dh, dv):
        # Adjoint of F
        dh_prev = F.pad(dh[:, :, :, :-1], (1, 0, 0, 0))
        h_adj = dh_prev - dh
        dv_prev = F.pad(dv[:, :, :-1, :], (0, 0, 1, 0))
        v_adj = dv_prev - dv
        return h_adj + v_adj

    def soft_thresh(x, th):
        return torch.sign(x) * torch.clamp(torch.abs(x) - th, min=0.0)

    for i in range(max_iter):
        # 1. Update t
        t_new = (1.0 / (2 * phi + 1)) * (f_gp + phi * (Q1 + H1 + Q3 + H3))

        # 2. Update Q1 (Unrolled gradient descent to approximate inverse of (I + F^T F))
        for _ in range(5):
            Fh_Q1, Fv_Q1 = apply_F(Q1)
            grad_Q1 = -gamma * (f_gp - Q1) - phi * (t_new - Q1 - H1) + phi * apply_FT(Fh_Q1 - Q2_h - H2_h, Fv_Q1 - Q2_v - H2_v)
            Q1 = Q1 - 0.2 * grad_Q1

        # 3. Update Q2 using Soft Thresholding for L1 spatial constraints
        Fh_Q1, Fv_Q1 = apply_F(Q1)
        Q2_h_new = soft_thresh(Fh_Q1 - H2_h, threshold)
        Q2_v_new = soft_thresh(Fv_Q1 - H2_v, threshold)

        # 4. Update Q3 Non-negativity constraint
        Q3_new = torch.clamp(t_new - H3, min=0.0)

        # 5. Update multipliers (H)
        H1_new = H1 - (t_new - Q1)
        H2_h_new = H2_h - (Fh_Q1 - Q2_h_new)
        H2_v_new = H2_v - (Fv_Q1 - Q2_v_new)
        H3_new = H3 - (t_new - Q3_new)

        if torch.norm(t_new - t) < 1e-4:
            t = t_new
            break

        t = t_new
        Q2_h, Q2_v, Q3 = Q2_h_new, Q2_v_new, Q3_new
        H1, H2_h, H2_v, H3 = H1_new, H2_h_new, H2_v_new, H3_new

    return t


# ==========================================
# 4. G-MDRF Pipeline & Input Adaptation
# ==========================================
class GMDRF_Pipeline(nn.Module):
    def __init__(self, train_loader, num_classes, spectral_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # --- Inducing Points Extraction (0.8% of training data) ---
        print("Extracting inducing points (~0.8% of training samples)...")
        all_center_pixels = []
        for data, _ in train_loader:
            # Extract center pixel: [batch_size, 1, 50, 9, 9] -> [batch_size, 50]
            center = data[:, 0, :, 4, 4]
            all_center_pixels.append(center)

        full_train_x = torch.cat(all_center_pixels, dim=0)
        num_inducing = max(int(0.008 * full_train_x.size(0)), 10)  # At least 10 points
        indices = torch.randperm(full_train_x.size(0))[:num_inducing]
        inducing_points = full_train_x[indices].to(device)

        # Initialize SIVGP
        self.gp_model = SIVGPModel(inducing_points, num_classes).to(device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_classes
        ).to(device)

    def train_step(self, x_patch, y, optimizer, mll):
        self.gp_model.train()
        self.likelihood.train()

        # Data Adaptation: Unwrap [B, 1, C, H, W] to center pixel [B, C]
        center_idx = x_patch.shape[-1] // 2
        x_center = x_patch[:, 0, :, center_idx, center_idx].float()

        optimizer.zero_grad()
        output = self.gp_model(x_center)

        # Convert classification labels to one-hot for GP regression approximation
        # Map 0, 1 to -1, 1 as it helps GPyTorch GP regression converge faster without vanishing
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float() * 2 - 1

        loss = -mll(output, y_one_hot)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.gp_model.eval()
        self.likelihood.eval()

        all_preds = []
        all_targets = []

        print("Running SIVGP + SAMRF Evaluation...")
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for data, target in test_loader:
                data, target = data.to(self.device).float(), target.to(self.device)

                # 1. Spectral Field (SIVGP) predicts for the whole 9x9 patch
                B = data.size(0)
                C_in = data.size(2)
                x_all = data[:, 0].permute(0, 2, 3, 1).reshape(B * 81, C_in)
                predictions = self.likelihood(self.gp_model(x_all))
                f_gp_all = predictions.mean  # [B*81, num_classes]
                f_gp = f_gp_all.view(B, 9, 9, self.num_classes).permute(0, 3, 1, 2)  # [B, num_classes, 9, 9]

                # 2. Spatial Field (SAMRF via ADMM)
                # Applying the spatial constraint over the continuous function space
                # Reduced delta_sp from 10.0 to 0.1 so soft-threshold doesn't collapse everything
                t_spatial = samrf_admm_patch(f_gp, delta_sp=0.1, phi=0.1)

                # Softmax to get final class (predictions are based on center pixel at index 4, 4)
                pred_class = torch.argmax(t_spatial[:, :, 4, 4], dim=1)

                all_preds.extend(pred_class.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate Metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        oa = accuracy_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)

        # Average Per-Class Accuracy (AA)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        aa = np.nanmean(per_class_acc)

        return oa, aa, kappa, per_class_acc


# ==========================================
# Execution Loop Configuration
# ==========================================
def run_gmdrf(train_loader, test_loader, num_classes, spectral_dim, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = GMDRF_Pipeline(train_loader, num_classes, spectral_dim, device)

    # Use Adafactor for the GP parameters as specified in the paper
    optimizer = Adafactor(
        [
            {"params": pipeline.gp_model.parameters()},
            {"params": pipeline.likelihood.parameters()},
        ],
        lr=5e-2,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    # Marginal Log Likelihood for Variational GP
    mll = gpytorch.mlls.VariationalELBO(
        pipeline.likelihood, pipeline.gp_model, num_data=len(train_loader.dataset)
    )

    print(f"Starting Adafactor Optimization for SIVGP...")
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            loss = pipeline.train_step(data, target, optimizer, mll)
            epoch_loss += loss

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f} | Time: {time.time() - start_time:.2f}s"
        )

    # Final Evaluation combining Spectral and Spatial fields
    oa, aa, kappa, per_class_acc = pipeline.evaluate(test_loader)

    CLASS_NAMES = [
        "Alfalfa", "Corn notill", "Corn mintill", "Corn",
        "Grass pasture", "Grass trees", "Grass pasture mowed", "Hay windrowed",
        "Oats", "Soybean notill", "Soybean mintill", "Soybean clean",
        "Wheat", "Woods", "Buildings Grass Trees Drives", "Stone Steel Towers",
    ]

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Overall Accuracy    (OA)  : {oa * 100:7.4f} %    │")
    print(f"  │  Average Accuracy    (AA)  : {aa * 100:7.4f} %    │")
    print(f"  │  Cohen's Kappa       (K)   : {kappa:10.6f}    │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"\n  Per-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        if np.isnan(acc): acc = 0.0
        bar = "█" * int(acc * 30)
        print(f"  {i + 1:2d}. {name:<35s} {acc * 100:6.2f}%  {bar}")


# ==========================================
# 5. Dataloaders & Main Execution
# ==========================================
PROCESSED_ROOT = "/home/23dcs505/datasets/IP"

class HyperspectralDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(dataset_abbr: str = "IP", pca_components: int = 50, batch_size: int = 64):
    proc_dir = os.path.join(PROCESSED_ROOT, f"pca_{pca_components}", dataset_abbr)
    if not os.path.isdir(proc_dir):
        # Fallback for local execution
        proc_dir = f"/home/Stark/Downloads/{dataset_abbr}_5"
    
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"Data directory not found: {proc_dir}")

    X_tr = torch.load(os.path.join(proc_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(proc_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(proc_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(proc_dir, "y_test.pt"))

    if y_tr.dtype == torch.uint8:
        y_tr = y_tr.long()
    if y_te.dtype == torch.uint8:
        y_te = y_te.long()

    # Apply Z-score Normalization feature-wise so GP lengthscales do not collapse
    # X_tr shape: [N, 1, 50, 9, 9] (Channels is dim 2)
    mean = X_tr.mean(dim=(0, 3, 4), keepdim=True)
    std = X_tr.std(dim=(0, 3, 4), keepdim=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    y_np = y_tr.numpy()
    num_classes = 16
    counts = np.bincount(y_np, minlength=num_classes).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    s_w = torch.from_numpy(weights[y_np])
    sampler = WeightedRandomSampler(s_w, len(s_w), replacement=True)

    train_loader = DataLoader(HyperspectralDataset(X_tr, y_tr), batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(HyperspectralDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    print("=" * 62)
    print("  G-MDRF — Indian Pines (PCA-50)")
    print("=" * 62)
    
    train_loader, test_loader = get_dataloaders(dataset_abbr="IP", pca_components=500, batch_size=64)
    run_gmdrf(train_loader, test_loader, num_classes=16, spectral_dim=50, epochs=500)
