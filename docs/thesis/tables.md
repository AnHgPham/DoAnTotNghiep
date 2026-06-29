# Thesis Tables

## Table 1 - Dataset Summary

| Dataset | Role | Split/Size | Notes |
|---|---|---:|---|
| MSWC Microset English | Main training/experiment anchor | 96,099 WAV total | Official CSV split, avoids leakage. |
| Google Speech Commands v2 | Evaluation/demo | 35 words | `gsc_edgespot_exact`, k-shot 10. |
| MSWC Top500 | Scale-up training | 450 train + 50 val words | Current local artifact is epoch13. |
| DEMAND | Noise augmentation | external noise clips | Not required for GSC evaluation. |

## Table 2 - Model Comparison

| Model | Feature | Loss | Params | Role |
|---|---|---|---:|---|
| DSCNN-L | MFCC | Triplet | small | baseline |
| EdgeSpotFull T4 | mel-PCEN | SCAF | ~130,598 | ablation |
| EdgeSpotFull T4 | mel-PCEN | SCAF+GE2E | ~130,598 | selected |

## Table 3 - Microset Result

| Configuration | ACC@5%FAR | KW-ACC | F1 | AUC | EER |
|---|---:|---:|---:|---:|---:|
| EdgeSpotFull T4 + SCAF | 85.21% | 74.52% | 81.92% | see result JSON | see result JSON |
| EdgeSpotFull T4 + SCAF+GE2E | 86.12% | 77.66% | 82.41% | 95.61% | 11.54% |

## Table 4 - Top500 Epoch13 Dev30

| Metric | Value |
|---|---:|
| ACC@1%FAR | 86.68% |
| ACC@5%FAR | 88.87% |
| FRR@5%FAR | 20.36% |
| AUC | 95.12% |
| F1 | 81.71% |

## Table 5 - Claim Matrix Summary

| Claim | Evidence | Report Use |
|---|---|---|
| Microset EdgeSpotFull T4 + SCAF+GE2E is the current thesis anchor. | Local checkpoint/result manifest. | Thesis main result. |
| Top500 epoch13 is available locally and promising. | Local checkpoint + dev30 result. | Demo/preliminary. |
| Top500 epoch25 is historical unless artifact is recovered. | Log/package manifest only. | Progress story, not final claim. |
| Open-set UI 17/17 is demo-level sampled evaluation. | UI/API result. | Demo explanation, not replacement for test100. |
