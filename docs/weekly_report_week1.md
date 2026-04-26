# [Internship 25-26] Enhanced Few-Shot Open-Set Keyword Spotting with Noise-Robust Prototype Classification and Real-Time Streaming Inference

**Student:** Phạm Hoàng An — 23BI14002
**Supervisor:** Dr. Tung
**Week 1 Report** — Sunday, [4/5/2026]

---

## Abstract

This week I finished setting up the project, wrote the part to extract MFCC from audio and model DSCNN-L. I find 1 folder for short word audio  "Google Speech Commands v2"
. Next week I will work on the Triplet Loss part.


## What I Have Done So Far (Week 1)

### 1. Project Setup
- Initialized project structure with organized directories: `src/features/`, `src/models/`, `configs/`, `data/`, `tests/`
- Created `requirements.txt` with all dependencies (PyTorch, torchaudio, numpy, etc.)
- Set up `configs/default.yaml` with all hyperparameters (audio settings, MFCC config, model architecture, training schedule)

### 2. MFCC Feature Extraction (`src/features/mfcc.py`)
- Implemented `MFCCExtractor` class using torchaudio
- Pipeline: raw audio (16 kHz, 1s) → 40 MFCC coefficients → narrow to first 10 → transpose → output shape (1, 47, 10)
- Parameters: n_fft=1024, hop_length=320, center=False, mel_scale='slaney', norm='slaney'

### 3. DSCNN-L Encoder (`src/models/dscnn.py`)
- Implemented Depthwise Separable CNN — Large variant (DSCNN-L)
- Architecture: 276 channels, 5 depthwise separable blocks, embedding_dim=276
- Input: (B, 1, 47, 10) → Output: (B, 276) L2-normalized embeddings
- Lightweight design suitable for on-device deployment

### 4. Data Download Script (`data/download_gsc.py`)
- Implemented automated download + extraction of Google Speech Commands v2 (~2.3 GB, 35 word classes)
- Includes progress bar and skip-if-exists logic

### 5. Unit Tests
- `tests/test_mfcc.py`: validates MFCC output shapes and coefficient count
- `tests/test_dscnn.py`: validates DSCNN input/output shapes, parameter count, embedding normalization

### 6. Documentation
- Prepared internship outline (Annex 1 format) and internship proposal documents

---

**No problems encountered this week.**

**Plan for next week:**
- Implement Triplet Loss and episodic training sampler (`src/models/prototypical.py`)
- Implement DEMAND noise augmentation (`src/features/augmentation.py`)
- Implement OpenNCM classifier with Direct L2 distance (`src/classifiers/open_ncm.py`)
- Begin setting up the training script
