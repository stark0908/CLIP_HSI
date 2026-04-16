# SpectralFormer Training Script

This is an exact replica of the official SpectralFormer implementation (`demo.py`) adapted for your local environment.

**Reference:** Hong et al., IEEE TGRS 2022  
**Official Repo:** https://github.com/danfenghong/IEEE_TGRS_SpectralFormer

## Data Requirements

The script expects `.mat` files in `./data/` directory:

- `IndianPine.mat` - Indian Pines dataset
- `Pavia.mat` - Pavia University dataset  
- `Houston.mat` - Houston dataset

Each `.mat` file should contain:
- `input` - HSI data (H × W × B)
- `TR` - Training set ground truth
- `TE` - Test set ground truth

**Download from:** https://drive.google.com/drive/folders/1nRphkwDZ74p-Al_O_X3feR24aRyEaJDY

## Usage

### Training ViT on Indian Pines

```bash
python train.py --dataset='Indian' --epoch=1400 --patches=1 --band_patches=1 --mode='ViT' --weight_decay=0
```

### Training Pixel-wise SpectralFormer

```bash
python train.py --dataset='Indian' --epoches=290 --patches=1 --band_patches=3 --mode='CAF' --weight_decay=0
```

### Training Patch-wise SpectralFormer

```bash
python train.py --dataset='Indian' --epoches=300 --patches=7 --band_patches=3 --mode='CAF' --weight_decay=5e-3
```

### Testing with Pretrained Model

```bash
python train.py --dataset='Indian' --flag_test=test --patches=7 --band_patches=3 --mode='CAF'
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | 'Indian' | Dataset: 'Indian', 'Pavia', or 'Houston' |
| `--flag_test` | str | 'train' | Mode: 'train' or 'test' |
| `--mode` | str | 'ViT' | Model: 'ViT' or 'CAF' (SpectralFormer) |
| `--patches` | int | 1 | Spatial patch size |
| `--band_patches` | int | 1 | Spectral band grouping factor |
| `--epoches` | int | 300 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size |
| `--learning_rate` | float | 5e-4 | Learning rate |
| `--weight_decay` | float | 0 | Weight decay |
| `--gamma` | float | 0.9 | Learning rate decay factor |
| `--test_freq` | int | 5 | Test evaluation frequency (epochs) |
| `--gpu_id` | str | '0' | GPU ID |
| `--seed` | int | 0 | Random seed |

## Key Features

✅ **Exact Official Implementation**
- Data loading from `.mat` files
- Band normalization (no PCA)
- Mirror padding for boundary handling
- Patch extraction with spectral band grouping
- ViT and CAF (Cross-layer Adaptive Fusion) modes

✅ **Data Pipeline**
1. Load raw HSI data
2. Normalize by spectral band
3. Extract train/test/true samples
4. Apply mirror padding
5. Extract spatial patches
6. Group spectral bands (gain_neighborhood_band)
7. Convert to PyTorch tensors

✅ **Training Features**
- Cross-entropy loss
- Adam optimizer with step-based LR decay
- Confusion matrix and metrics (OA, AA, Kappa)
- Comprehensive logging

## Output Files

- `training.log` - Training logs
- `model_<dataset>_<mode>.pt` - Saved model weights
- `classification_map.mat` - Predicted classification map (in test mode)

## Recommended Settings

### Indian Pines
- **ViT:** `--epoches=1400 --patches=1 --band_patches=1 --mode='ViT' --weight_decay=0`
- **Pixel-wise:** `--epoches=290 --patches=1 --band_patches=3 --mode='CAF' --weight_decay=0`
- **Patch-wise:** `--epoches=300 --patches=7 --band_patches=3 --mode='CAF' --weight_decay=5e-3`

### Pavia University
- **ViT:** `--epoches=1000 --patches=1 --band_patches=1 --mode='ViT' --weight_decay=0`
- **Pixel-wise:** `--epoches=500 --patches=1 --band_patches=3 --mode='CAF' --weight_decay=5e-3`
- **Patch-wise:** `--epoches=480 --patches=7 --band_patches=7 --mode='CAF' --weight_decay=5e-3`

### Houston
- **ViT:** `--epoches=900 --patches=1 --band_patches=1 --mode='ViT' --weight_decay=0`
- **Pixel-wise:** `--epoches=520 --patches=1 --band_patches=3 --mode='CAF' --weight_decay=5e-3`
- **Patch-wise:** `--epoches=600 --patches=7 --band_patches=3 --mode='CAF' --weight_decay=5e-3`

## Model Architecture

### ViT (Vision Transformer)
- Standard Vision Transformer baseline
- Multi-head self-attention
- Feed-forward networks with GELU activation

### CAF (SpectralFormer with Cross-layer Adaptive Fusion)
- Enhanced ViT with skip connections
- Adaptive fusion of layer outputs
- Better spectral feature learning

## Performance Notes

- Due to parameter initialization randomness, results may slightly differ from the paper
- Use the same seed for reproducibility
- For large datasets (Pavia, Houston), consider splitting into sub-images if memory issues occur

## License

This implementation follows the GNU General Public License v3.0 of the original repository.

**Citation:**
```bibtex
@article{hong2022spectralformer,
  title={Spectralformer: Rethinking hyperspectral image classification with transformers},
  author={Hong, Danfeng and Han, Zhu and Yao, Jing and Gao, Lianru and Zhang, Bing and Plaza, Antonio and Chanussot, Jocelyn},
  journal={IEEE Trans. Geosci. Remote Sens.},
  year={2022},
  volume={60},
  pages={1-15},
  note = {DOI: 10.1109/TGRS.2021.3130716}
}
```
