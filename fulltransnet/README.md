# FullTransNet: Full Transformer with Local-Global Attention for Video Summarization

A pure PyTorch implementation of the FullTransNet paper (Lan et al., 2024) for video summarization.

## Key Features

- **Local-Global Attention**: Sliding-window local attention + global attention on change-point frames
- **Encoder-Decoder Transformer**: 6-layer encoder with LGA + 6-layer decoder with standard MHA
- **Pure PyTorch**: No dependency on Longformer or custom CUDA kernels — runs on any platform
- **TVSum Dataset**: Configured to use the TVSum dataset from the parent `data/` folder

## Project Structure

```
fulltransnet/
├── train.py              # Main training entry point
├── evaluate.py           # Evaluation script
├── make_split.py         # Generate train/test splits
├── make_shots.py         # Compute shot boundaries (KTS)
├── requirements.txt      # Python dependencies
├── model/
│   ├── attention.py      # Local-Global Attention (pure PyTorch)
│   ├── transformer.py    # Encoder-Decoder Transformer
│   ├── losses.py         # Loss functions
│   └── train_loop.py     # Training loop per split
├── helpers/
│   ├── data_helper.py    # Dataset loading & utilities
│   ├── vsumm_helper.py   # Video summary evaluation (F1, knapsack)
│   └── init_helper.py    # Argument parsing & initialization
├── kts/
│   ├── cpd_auto.py       # Auto change-point detection
│   └── cpd_nonlin.py     # Non-linear change-point detection
└── splits/
    └── tvsum.yml          # 5-fold cross-validation splits
```

## Setup

```bash
pip install -r requirements.txt
```

The TVSum dataset (`eccv16_dataset_tvsum_google_pool5.h5`) should be in `../data/` relative to this folder.

## Training

Train on TVSum with default settings (300 epochs, BCE loss):

```bash
python train.py
```

Custom training:

```bash
python train.py --max-epoch 100 --lr 0.0013 --loss bce --device cuda
python train.py --device cpu --max-epoch 10   # Quick test on CPU
```

## Evaluation

Evaluate saved checkpoints:

```bash
python evaluate.py --model-dir ./model_save/tvsum --splits ./splits/tvsum.yml
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Input dim | 1024 (GoogLeNet pool5) |
| Hidden dim | 64 |
| Attention heads | 8 |
| Encoder layers | 6 (Local-Global Attention) |
| Decoder layers | 6 (Standard Multi-Head Attention) |
| Window size | 16 |
| FFN dim | 2048 |
| Max sequence length | 1536 |

## Acknowledgments

Based on the [FullTransNet](https://github.com/ChiangLu/FullTransNet) paper and official implementation.
Uses datasets from [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) and pipeline from [VASNet](https://github.com/ok1zjf/VASNet).
