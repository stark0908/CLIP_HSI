import os
import time
import json
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from einops import rearrange
from operator import truediv

# ==========================================
# 0. CONFIG
# ==========================================
BATCH_SIZE_TRAIN = 64
EPOCHS = 100
LR = 0.001
NUM_CLASS = 16

# ==========================================
# 1. MODEL DEFINITION (from SSFTTnet.py)
# ==========================================
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
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
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*48, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x

# ==========================================
# 2. DATA PREPROCESSING & DATASET
# ==========================================
class HyperspectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float() if X.dtype != torch.float32 else X
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loader():
    data_dir = '/home/Stark/Downloads/IP_5'
    print(f"Loading datasets from {data_dir}...")
    
    X_tr = torch.load(os.path.join(data_dir, "X_train.pt"))
    y_tr = torch.load(os.path.join(data_dir, "y_train.pt"))
    X_te = torch.load(os.path.join(data_dir, "X_test.pt"))
    y_te = torch.load(os.path.join(data_dir, "y_test.pt"))

    print(f"Train data shape: {X_tr.shape}, Train labels shape: {y_tr.shape}")
    print(f"Test data shape: {X_te.shape}, Test labels shape: {y_te.shape}")

    # Convert y_train to numpy for class counting
    y_np = y_tr.numpy() if isinstance(y_tr, torch.Tensor) else y_tr
    class_counts = np.bincount(y_np.astype(int), minlength=NUM_CLASS).astype(np.float32)
    class_w = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_w = torch.tensor(class_w[y_np.astype(int)], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    nw = min(4, (os.cpu_count() or 2) // 2) # multiprocessing workers
    pin_memory = torch.cuda.is_available()

    trainset = HyperspectralDataset(X_tr, y_tr)
    testset = HyperspectralDataset(X_te, y_te)

    train_loader = DataLoader(dataset=trainset,
                              batch_size=BATCH_SIZE_TRAIN,
                              sampler=sampler,
                              num_workers=nw,
                              pin_memory=pin_memory,
                              drop_last=True)
                              
    test_loader = DataLoader(dataset=testset,
                             batch_size=BATCH_SIZE_TRAIN,
                             shuffle=False,
                             num_workers=nw,
                             pin_memory=pin_memory)
                             
    return train_loader, test_loader, None, None

# ==========================================
# 4. PLOTTING MAPS (from get_cls_map.py)
# ==========================================
def get_classification_map(y_pred, y):
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1
    return cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    colors = {
        0: [0, 0, 0],       1: [147, 67, 46],   2: [0, 0, 255],     3: [255, 100, 0],
        4: [0, 255, 123],   5: [164, 75, 155],  6: [101, 174, 255], 7: [118, 254, 172],
        8: [60, 91, 112],   9: [255, 255, 0],   10: [255, 255, 125],11: [255, 0, 255],
        12: [100, 0, 255],  13: [0, 172, 254],  14: [0, 255, 0],    15: [171, 175, 80],
        16: [101, 193, 60]
    }
    for index, item in enumerate(x_list):
        if item in colors:
            y[index] = np.array(colors[item]) / 255.
    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    plt.close()

def get_cls_map_func(net, device, all_data_loader, y):
    y_pred, _ = test_inference(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    
    os.makedirs('classification_maps', exist_ok=True)
    classification_map(y_re, y, 300, 'classification_maps/IP_predictions.eps')
    classification_map(y_re, y, 300, 'classification_maps/IP_predictions.png')
    classification_map(gt_re, y, 300, 'classification_maps/IP_gt.png')
    print('------Get classification maps successful-------')

# ==========================================
# 5. TRAINING AND EVALUATION (from IP_train.py)
# ==========================================
def test_inference(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels.numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels.numpy()))
    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                    'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

def train(train_loader, test_loader, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SSFTTnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    best_test_loss = float("inf")
    os.makedirs('cls_params', exist_ok=True)
    best_model_path = 'cls_params/SSFTTnet_params.pth'
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        tr_loss = tr_correct = tr_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            tr_correct += outputs.detach().argmax(1).eq(target).sum().item()
            tr_total += target.size(0)
            
        avg_tr = tr_loss / len(train_loader)
        tr_acc = 100.0 * tr_correct / tr_total
        
        net.eval()
        te_loss = te_correct = te_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = net(data)
                loss = criterion(outputs, target)
                te_loss += loss.item()
                te_correct += outputs.argmax(1).eq(target).sum().item()
                te_total += target.size(0)
                
        avg_te = te_loss / len(test_loader)
        te_acc = 100.0 * te_correct / te_total
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"TrLoss {avg_tr:.4f}  TrAcc {tr_acc:.2f}% | "
              f"TeLoss {avg_te:.4f}  TeAcc {te_acc:.2f}% | {elapsed:.1f}s")
        
        if avg_te < best_test_loss:
            best_test_loss = avg_te
            torch.save(net.state_dict(), best_model_path)
            print(f"  ★ Best model saved  (loss={best_test_loss:.4f})")
            
    print('Finished Training')
    return net, device

if __name__ == '__main__':
    train_loader, test_loader, all_data_loader, y_all = create_data_loader()
    
    tic1 = time.perf_counter()
    net, device = train(train_loader, test_loader, epochs=EPOCHS)
    toc1 = time.perf_counter()
    
    # Load the best model configuration before evaluating
    net.load_state_dict(torch.load('cls_params/SSFTTnet_params.pth'))
    
    tic2 = time.perf_counter()
    y_pred_test, y_test = test_inference(device, net, test_loader)
    toc2 = time.perf_counter()
    
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification_str = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    
    os.makedirs('cls_result', exist_ok=True)
    file_name = "cls_result/classification_report.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)\n'.format(Training_Time))
        x_file.write('{} Test_time (s)\n'.format(Test_time))
        x_file.write('{} Kappa accuracy (%)\n'.format(kappa))
        x_file.write('{} Overall accuracy (%)\n'.format(oa))
        x_file.write('{} Average accuracy (%)\n'.format(aa))
        x_file.write('{} Each accuracy (%)\n'.format(each_acc))
        x_file.write('{}\n'.format(classification_str))
        x_file.write('{}\n'.format(confusion))

    CLASS_NAMES = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                   'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                   'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                   'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                   'Stone-Steel-Towers']

    print(f"\n{'─' * 42}")
    print(f"  Overall Accuracy  (OA) : {oa:.2f}%")
    print(f"  Average Accuracy  (AA) : {aa:.2f}%")
    print(f"  Kappa Coefficient  (κ) : {kappa / 100.0:.4f}")
    print(f"{'─' * 42}")
    print("\nPer-Class Accuracies:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, each_acc)):
        print(f"  {i + 1:2d}. {name:<35s}: {acc:.2f}%")

    if all_data_loader is not None and y_all is not None:
        get_cls_map_func(net, device, all_data_loader, y_all)
    else:
        print('\nSkipped classification map generation as the full unpatched dataset was not provided.')
