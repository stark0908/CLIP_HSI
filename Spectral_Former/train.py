#!/usr/bin/env python3
"""
SpectralFormer Training Script
Exact replica of official demo.py adapted for local environment
Reference: Hong et al., IEEE TGRS 2022
Official repo: github.com/danfenghong/IEEE_TGRS_SpectralFormer
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING UTILITIES (from official demo.py)
# ═══════════════════════════════════════════════════════════════════════════

def mirror_hsi(height, width, band, input_normalize, patch=5):
    """Mirror padding for HSI data"""
    padding = patch // 2
    mirror_hsi_data = np.zeros((height + 2*padding, width + 2*padding, band), dtype=float)
    
    # Center region
    mirror_hsi_data[padding:(padding+height), padding:(padding+width), :] = input_normalize
    
    # Left mirror
    for i in range(padding):
        mirror_hsi_data[padding:(height+padding), i, :] = input_normalize[:, padding-i-1, :]
    
    # Right mirror
    for i in range(padding):
        mirror_hsi_data[padding:(height+padding), width+padding+i, :] = input_normalize[:, width-1-i, :]
    
    # Top mirror
    for i in range(padding):
        mirror_hsi_data[i, :, :] = mirror_hsi_data[padding*2-i-1, :, :]
    
    # Bottom mirror
    for i in range(padding):
        mirror_hsi_data[height+padding+i, :, :] = mirror_hsi_data[height+padding-1-i, :, :]
    
    return mirror_hsi_data


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    """Extract spatial patch around a pixel"""
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x+patch), y:(y+patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    """Group neighboring spectral bands (band patches)"""
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band), dtype=float)
    
    # Center region
    x_train_band[:, nn*patch*patch:(nn+1)*patch*patch, :] = x_train_reshape
    
    # Left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, :i+1] = x_train_reshape[:, :, band-i-1:]
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, i+1:] = x_train_reshape[:, :, :band-i-1]
        else:
            x_train_band[:, i:(i+1), :i+1] = x_train_reshape[:, 0:1, band-i-1:]
            x_train_band[:, i:(i+1), i+1:] = x_train_reshape[:, 0:1, :band-i-1]
    
    # Right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, :band-i-1] = x_train_reshape[:, :, i+1:]
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, band-i-1:] = x_train_reshape[:, :, :i+1]
        else:
            x_train_band[:, (nn+1+i):(nn+2+i), :band-i-1] = x_train_reshape[:, 0:1, i+1:]
            x_train_band[:, (nn+1+i):(nn+2+i), band-i-1:] = x_train_reshape[:, 0:1, :i+1]
    
    return x_train_band


def choose_train_test_point(train_data, test_data, true_data, num_classes):
    """Extract pixel coordinates for train/test/true sets"""
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    
    # Training data
    for i in range(num_classes):
        each_class = np.argwhere(train_data == (i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    
    # Test data
    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
    
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    
    # True data
    for i in range(num_classes+1):
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class
    
    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def train_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    """Extract patches for train/test/true sets"""
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    
    return x_train_band, x_test_band, x_true_band


def train_test_label(number_train, number_test, number_true, num_classes):
    """Create labels for train/test/true sets"""
    y_train = []
    y_test = []
    y_true = []
    
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    
    return np.array(y_train), np.array(y_test), np.array(y_true)


# ═══════════════════════════════════════════════════════════════════════════
# VISIONTRANSFORMER MODEL (from official vit_pytorch.py)
# ═══════════════════════════════════════════════════════════════════════════

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: (t.view(b, n, h, -1).transpose(1, 2)), qkv)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode='ViT'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))
        
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))
    
    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1
        
        return x


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, 
                 pool='cls', channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()
        
        patch_dim = image_size ** 2 * near_band
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x, mask)
        
        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Calculate accuracy@k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


def cal_results(matrix):
    """Calculate OA, AA, Kappa from confusion matrix"""
    shape = np.shape(matrix)
    number = 0
    sum_val = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum_val += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum_val / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    
    return OA, AA_mean, Kappa, AA


def output_metric(tar, pre):
    """Calculate metrics"""
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(DEVICE)
        batch_target = batch_target.to(DEVICE)
        
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion, optimizer):
    """Validate model"""
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(DEVICE)
        batch_target = batch_target.to(DEVICE)
        
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    
    return tar, pre


def test_epoch(model, test_loader, criterion, optimizer):
    """Test model"""
    pre = np.array([])
    
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(DEVICE)
        batch_target = batch_target.to(DEVICE)
        
        batch_pred = model(batch_data)
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    
    return pre


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    """Main training function"""
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    logger.info("=" * 80)
    logger.info("SpectralFormer Training - Official Repo Implementation")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Patches: {args.patches}, Band Patches: {args.band_patches}")
    logger.info(f"Epochs: {args.epoches}, Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    
    # Load data
    logger.info("\nLoading data...")
    data_dir = "/home/23dcs505/datasets"
    try:
        if args.dataset == 'Indian':
            data = loadmat(os.path.join(data_dir, 'IndianPine.mat'))
        elif args.dataset == 'Pavia':
            data = loadmat(os.path.join(data_dir, 'Pavia.mat'))
        elif args.dataset == 'Houston':
            data = loadmat(os.path.join(data_dir, 'Houston.mat'))
        else:
            raise ValueError("Unknown dataset")
    except FileNotFoundError:
        logger.error(f"Data files not found in {data_dir}")
        logger.error("Expected: IndianPine.mat, Pavia.mat, or Houston.mat")
        logger.error("Download from: https://drive.google.com/drive/folders/1nRphkwDZ74p-Al_O_X3feR24aRyEaJDY")
        return
    
    TR = data['TR']
    TE = data['TE']
    input_data = data['input']
    label = TR + TE
    num_classes = np.max(TR)
    
    height, width, band = input_data.shape
    logger.info(f"Data shape: height={height}, width={width}, band={band}")
    logger.info(f"Num classes: {num_classes}")
    
    # Normalize data by band
    logger.info("Normalizing data...")
    input_normalize = np.zeros(input_data.shape)
    for i in range(input_data.shape[2]):
        input_max = np.max(input_data[:, :, i])
        input_min = np.min(input_data[:, :, i])
        input_normalize[:, :, i] = (input_data[:, :, i] - input_min) / (input_max - input_min)
    
    # Extract train/test/true points
    logger.info("Extracting train/test/true points...")
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = \
        choose_train_test_point(TR, TE, label, num_classes)
    
    logger.info(f"Train samples: {total_pos_train.shape[0]}")
    logger.info(f"Test samples: {total_pos_test.shape[0]}")
    logger.info(f"True samples: {total_pos_true.shape[0]}")
    
    # Mirror HSI
    logger.info("Applying mirror padding...")
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    
    # Extract patches with band grouping
    logger.info("Extracting patches...")
    x_train_band, x_test_band, x_true_band = train_test_data(
        mirror_image, band, total_pos_train, total_pos_test, total_pos_true,
        patch=args.patches, band_patch=args.band_patches
    )
    logger.info(f"x_train shape: {x_train_band.shape}")
    logger.info(f"x_test shape: {x_test_band.shape}")
    logger.info(f"x_true shape: {x_true_band.shape}")
    
    # Generate labels
    logger.info("Generating labels...")
    y_train, y_test, y_true = train_test_label(number_train, number_test, number_true, num_classes)
    
    # Convert to torch tensors
    logger.info("Converting to PyTorch tensors...")
    x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    Label_train = Data.TensorDataset(x_train, y_train)
    
    x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Label_test = Data.TensorDataset(x_test, y_test)
    
    x_true = torch.from_numpy(x_true_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_true = torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true = Data.TensorDataset(x_true, y_true)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
    label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)
    
    logger.info(f"Train batches: {len(label_train_loader)}")
    logger.info(f"Test batches: {len(label_test_loader)}")
    logger.info(f"True batches: {len(label_true_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = ViT(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=band,
        num_classes=num_classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params}, Trainable: {trainable_params}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    
    # Training/Testing
    if args.flag_test == 'test':
        logger.info("\nLoading pretrained model for testing...")
        try:
            if args.mode == 'ViT':
                model.load_state_dict(torch.load('./ViT.pt', map_location=DEVICE))
            elif (args.mode == 'CAF') and (args.patches == 1):
                model.load_state_dict(torch.load('./SpectralFormer_pixel.pt', map_location=DEVICE))
            elif (args.mode == 'CAF') and (args.patches == 7):
                model.load_state_dict(torch.load('./SpectralFormer_patch.pt', map_location=DEVICE))
            else:
                raise ValueError("Wrong parameters for loading model")
        except FileNotFoundError:
            logger.error("Model file not found!")
            return
        
        model.eval()
        logger.info("Testing...")
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        
        logger.info("\n" + "="*80)
        logger.info("Test Results:")
        logger.info(f"OA: {OA2:.4f} | AA: {AA_mean2:.4f} | Kappa: {Kappa2:.4f}")
        logger.info("="*80)
        logger.info(f"AA per class:\n{AA2}")
        
        # Generate classification map
        logger.info("Generating classification map...")
        pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1
        
        savemat('./classification_map.mat', {'prediction': prediction_matrix, 'label': label})
        logger.info("Classification map saved to classification_map.mat")
        
    elif args.flag_test == 'train':
        logger.info("\nStarting training...")
        tic = time.time()
        
        for epoch in range(args.epoches):
            scheduler.step()
            
            # Train
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            
            # Test
            if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                
                logger.info(f"Epoch {epoch+1}/{args.epoches} - "
                           f"Train Loss: {train_obj:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Test OA: {OA2:.4f}, Test AA: {AA_mean2:.4f}, Test Kappa: {Kappa2:.4f}")
        
        toc = time.time()
        logger.info(f"\nRunning time: {toc-tic:.2f}s")
        logger.info("\n" + "="*80)
        logger.info("Final Results:")
        logger.info(f"OA: {OA2:.4f} | AA: {AA_mean2:.4f} | Kappa: {Kappa2:.4f}")
        logger.info("="*80)
        logger.info(f"AA per class:\n{AA2}")
        
        # Save model
        torch.save(model.state_dict(), f'./model_{args.dataset}_{args.mode}.pt')
        logger.info(f"Model saved to ./model_{args.dataset}_{args.mode}.pt")


def print_args(args):
    """Print arguments"""
    for k, v in zip(args.keys(), args.values()):
        logger.info(f"{k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpectralFormer Training - Official Implementation")
    
    # Dataset
    parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], 
                       default='Indian', help='dataset to use')
    parser.add_argument('--flag_test', choices=['test', 'train'], 
                       default='train', help='testing or training mode')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], 
                       default='ViT', help='mode choice (ViT or CAF)')
    
    # GPU and seed
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_freq', type=int, default=5, help='test frequency (epochs)')
    parser.add_argument('--patches', type=int, default=1, help='spatial patch size')
    parser.add_argument('--band_patches', type=int, default=1, help='spectral band patches')
    parser.add_argument('--epoches', type=int, default=300, help='total epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    logger.info(f"\n{'='*80}")
    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*80}\n")
    
    # Run
    main(args)
