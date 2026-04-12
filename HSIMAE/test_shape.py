import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    SPECTRAL_DIM = 50; PATCH_H = 9; PATCH_W = 9; IN_CHANS = 1; NUM_CLASSES = 16
    SPATIAL_STRIDE = 3; SPECTRAL_STRIDE = 10
    N_SPATIAL_TOKENS = 9; N_SPECTRAL_TOKENS = 5; N_TOKENS = 45
    TOKEN_DIM = 128; SSSE_DEPTH = 3; FUSION_DEPTH = 1; N_HEADS = 8
    DECODER_DIM = 64; DECODER_DEPTH = 2; DECODER_HEADS = 8
    MASK_RATIO_SPA = 0.60; MASK_RATIO_SPE = 0.50
    MIN_SPA_KEEP = 2; MIN_SPE_KEEP = 2
    LAMBDA_REC = 10.0

cfg = Config()

from HSIMAE import HSIMAEFinetune

model = HSIMAEFinetune(cfg)
x = torch.randn(2, 1, 50, 9, 9)
try:
    loss, _, _ = model(x, labels=torch.tensor([0, 1]))
    print("Success! Loss computed.")
except Exception as e:
    print(f"Error: {e}")
