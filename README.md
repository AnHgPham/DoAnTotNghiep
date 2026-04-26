# Enhanced Few-Shot Open-Set Keyword Spotting

Few-Shot Open-Set Keyword Spotting system with Direct L2 distance prototype classification, noise robustness, and real-time streaming inference.

## Features

- **Few-Shot Learning**: Learn new keywords from 3-5 audio samples without retraining
- **Open-Set Recognition**: Reject unknown words using acceptance radius threshold
- **Noise Robustness**: Noise augmentation during training + optional denoising at inference (EXT-1)
- **Streaming Inference**: VAD + sliding window for continuous real-time detection (EXT-2)
- **Speaker Verification**: Optional ECAPA-TDNN speaker gate (Optional Extension)
- **Multiple Loss Functions**: Triplet Loss, ArcFace, Sub-center ArcFace (SCAF)

## Setup

```bash
pip install -r requirements.txt
```

## Download Data

```bash
python data/download_gsc.py
python data/download_mswc.py
python data/convert_opus.py
```

## Project Structure

```
src/
├── features/       # MFCC extraction + noise augmentation + PCEN + SpecAugment
├── models/         # DSCNN-L encoder + Triplet/ArcFace/SCAF loss
├── classifiers/    # OpenNCM classifier (Direct L2 distance)
├── evaluation/     # Metrics (DET, AUC, EER, P/R/F1) + protocols (GSC, MSWC)
├── streaming/      # Silero VAD + sliding window engine (EXT-2)
├── enhancements/   # Denoiser (EXT-1) + Speaker verification (ECAPA-TDNN)
├── data/           # Dataset loaders (MSWC, GSC)
└── demo/           # Gradio web app
scripts/
├── train.py        # Training with validation + TensorBoard
├── evaluate.py     # Evaluation on GSC/MSWC protocols
└── compare_kshot.py # K-shot comparison utility
tests/              # Unit tests for all modules
notebooks/          # Colab training notebooks
configs/            # YAML configuration
docs/               # Thesis documents, proposals, reports
```

## Training

```bash
# Triplet Loss (default)
python scripts/train.py --config configs/default.yaml --data-dir data/gsc_v2

# ArcFace
python scripts/train.py --config configs/default.yaml --data-dir data/gsc_v2 --loss arcface

# Sub-center ArcFace
python scripts/train.py --config configs/default.yaml --data-dir data/gsc_v2 --loss scaf

# Resume training
python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pt
```

## Evaluation

```bash
# GSC Fixed protocol
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# GSC Random protocol
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --protocol gsc_random

# MSWC evaluation
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --protocol mswc_random
```

## Demo

```bash
python src/demo/app.py
```

Opens a Gradio web app at `http://localhost:7860` with:
1. **Offline Detection** - Upload/record audio, detect keywords
2. **Enrollment** - Register new keywords with 3-5 samples
3. **Streaming + VAD** - Test sliding-window detection with Silero VAD
4. **Settings** - Configure denoising, speaker gate, view model info

## Testing

```bash
pytest tests/ -v
```

## Architecture

- **Encoder**: DSCNN-L (276 channels, 5 DS blocks, embedding_dim=276)
- **Features**: MFCC (40 computed, 10 used, n_fft=1024, center=False) -> input shape (B, 1, 47, 10)
- **Training**: Episodic batching, StepLR(step_size=20, gamma=0.5), validation every 5 epochs
- **Loss**: Triplet (margin=0.5) | ArcFace (s=30, m=0.5) | SCAF (K=3)
- **Classification**: Direct L2 distance with acceptance radius threshold
- **Metrics**: AUC, EER, FRR@FAR, ACC@FAR, Precision, Recall, F1, DET curves
- **Datasets**: GSC v2 (eval), MSWC English (train), DEMAND (noise)
- **Extensions**: Spectral-gate denoising (EXT-1), Silero VAD streaming (EXT-2), ECAPA-TDNN speaker verification (Optional)

## Colab Training

For GPU training, use the Colab notebooks in `notebooks/`:
1. `01_train_colab.ipynb` - Basic training
2. `02_train_enhanced.ipynb` - Full pipeline with all experiments, evaluation, TensorBoard, and demo
