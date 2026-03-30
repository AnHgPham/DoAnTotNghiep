# Enhanced Few-Shot Open-Set Keyword Spotting

Few-Shot Open-Set Keyword Spotting system with Direct L2 distance prototype classification, noise robustness, and real-time streaming inference.

## Features

- **Few-Shot Learning**: Learn new keywords from 3-5 audio samples without retraining
- **Open-Set Recognition**: Reject unknown words using acceptance radius threshold
- **Noise Robustness**: Noise augmentation during training + optional denoising at inference
- **Streaming Inference**: VAD + sliding window for continuous real-time detection

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
├── features/       # MFCC extraction + noise augmentation
├── models/         # DSCNN-L encoder + Triplet Loss training
├── classifiers/    # OpenNCM classifier (Direct L2 distance)
├── evaluation/     # Metrics (DET, AUC, FAR/FRR) + protocols
├── streaming/      # VAD + sliding window engine
├── enhancements/   # Denoiser + speaker verification (optional)
└── demo/           # Gradio web app (optional)
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Testing

```bash
pytest tests/ -v
```

## Architecture

- **Encoder**: DSCNN-L (276 channels, 5 DS blocks, embedding_dim=276)
- **Features**: MFCC (40 computed, 10 used) -> input shape (B, 1, 49, 10)
- **Training**: Triplet Loss, episodic batching, StepLR(step_size=20, gamma=0.5)
- **Classification**: Direct L2 distance with acceptance radius threshold
- **Datasets**: GSC v2 (eval), MSWC English (train), DEMAND (noise)
