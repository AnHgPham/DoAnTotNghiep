# Thesis Draft EN - Few-Shot Open-Set Keyword Spotting

## Abstract

This project studies **few-shot open-set keyword spotting**. The system must recognize user-enrolled keywords from only a small number of examples while rejecting non-enrolled words as `unknown`. Unlike closed-set KWS, the model cannot be forced to always choose one class.

The original pipeline used DSCNN-L, MFCC features, and Triplet loss. Through MSWC Microset experiments, the project moved to an EdgeSpot-style encoder: **EdgeSpotFull T4**, mel-PCEN input, 64-D embeddings, and a hybrid **SCAF+GE2E** objective. SCAF improves class separation in embedding space; GE2E makes training closer to support-query prototype inference. This configuration is the current thesis anchor and the basis for scaling to MSWC Top500.

The locked Microset result reaches `ACC@5%FAR = 86.12%`, `KW-ACC = 77.66%`, `F1 = 82.41%`, and `AUC = 95.61%` on GSC test100. Top500 currently has a reliable local epoch13 checkpoint with promising GSC-dev results (`ACC@5%FAR = 88.87%`), but it is treated as preliminary because the Colab run was interrupted by session/unit limits. The web demo includes model switching, long-audio timing analysis, sampled open-set testing, and calibration.

## 1. Introduction

Keyword spotting detects target words in audio. Many practical systems use a fixed vocabulary and require many training examples per keyword. This project targets a more flexible setting: users can add custom keywords using only a few examples, and the system must reject unknown words rather than misclassifying them.

Project objectives:

1. Few-shot personalization.
2. Open-set rejection.
3. Explainable demo behavior for accepted, rejected, and missed detections.

## 2. Problem Statement

Given enrolled keywords `K = {k1, k2, ..., kn}` and a query audio `x`, the system should output either one keyword `ki` or `unknown`. This requires robust embeddings, calibrated thresholds, and a decision rule that does not over-accept unknown speech.

Main challenges:

- short and phonetically similar words;
- cross-dataset speaker/accent/noise shift;
- few enrollment examples;
- threshold tradeoff between false accept and false reject;
- long-audio segmentation errors;
- streaming window alignment and cooldown.

## 3. Research Questions

1. Does an EdgeSpot-style mel-PCEN encoder improve over DSCNN-L/MFCC/Triplet?
2. How do SCAF and GE2E affect prototype-based few-shot inference?
3. Can the Microset-selected configuration scale to Top500?
4. How do threshold, per-class threshold, and close-word guard affect open-set behavior?

## 4. Related Work

Closed-set KWS models such as CNN, DSCNN, and TC-ResNet are efficient but do not naturally reject unknown words. Few-shot methods instead learn an embedding space and classify by prototypes. EdgeSpot-style models provide compact KWS encoders. SCAF improves angular separation, while GE2E trains support/query centroid behavior originally popularized in verification tasks.

The project uses an EdgeSpot-style model but does not claim full EdgeSpot paper reproduction. GE2E is an added project component, not an original EdgeSpot component.

## 5. Proposed Method

Audio is resampled to 16 kHz mono, trimmed/padded to one second, converted to mel-PCEN features of shape `(1, 40, 101)`, and passed through EdgeSpotFull T4 to obtain a 64-D embedding.

Training uses:

```text
L = L_scaf + L_ge2e
```

Inference uses prototype means and L2 distance:

```text
prototype(keyword) = mean(support embeddings)
top1 = nearest prototype by L2
margin = distance(top2) - distance(top1)
accept if distance(top1) <= threshold and margin >= accept_margin
```

The current open-set demo recommendation is Guard ON, Per-class OFF, accept margin 0.05.

## 6. Dataset And Protocol

### MSWC Microset English

Microset is the main thesis experiment. It uses official CSV manifests to avoid leakage:

- train: about 69,868 WAV;
- dev: about 13,114 WAV;
- test: about 13,117 WAV.

### Google Speech Commands v2

GSC is used for cross-dataset evaluation and demo. The main protocol is `gsc_edgespot_exact`, k-shot 10, with true silence and unknown negative words.

### Top500

Top500 is the scale-up path:

- 450 train words;
- 50 validation words;
- full clips with `max_per_word=0`.

The reliable local artifact is currently epoch13. An earlier epoch25 run should only be mentioned as historical progress if complete local checkpoint/result artifacts are absent.

## 7. Experimental Setup

Compared configurations:

| Model | Feature | Loss | Role |
|---|---|---|---|
| DSCNN-L | MFCC | Triplet | baseline |
| EdgeSpotFull T4 | mel-PCEN | SCAF | ablation |
| EdgeSpotFull T4 | mel-PCEN | SCAF+GE2E | selected |

Evaluation metrics include ACC@1%FAR, ACC@5%FAR, FRR@5%FAR, AUC, EER, keyword accuracy, and F1.

## 8. Results

Microset locked result:

| Configuration | ACC@5%FAR | KW-ACC | F1 | AUC | EER |
|---|---:|---:|---:|---:|---:|
| EdgeSpotFull T4 + SCAF | 85.21% | 74.52% | 81.92% | available in logs | available in logs |
| EdgeSpotFull T4 + SCAF+GE2E | **86.12%** | **77.66%** | **82.41%** | **95.61%** | **11.54%** |

Top500 epoch13 GSC-dev:

| Metric | Value |
|---|---:|
| ACC@1%FAR | 86.68% |
| ACC@5%FAR | 88.87% |
| FRR@5%FAR | 20.36% |
| AUC | 95.12% |
| F1 | 81.71% |

Top500 is promising but preliminary because the available local checkpoint is epoch13 from an interrupted run.

## 9. Demo System

The web demo supports:

- model switching between Microset and Top500;
- GSC enrollment presets;
- single audio detection;
- long audio with labels and timing JSON;
- open-set 17 known / 17 unknown testing;
- calibration;
- streaming microphone detection;
- session report export.

## 10. Discussion

SCAF+GE2E improves the match between training and inference. SCAF pushes classes apart, while GE2E trains support/query centroid behavior. This combination is well aligned with few-shot prototype matching.

Open-set remains difficult. Guard OFF can raise known keyword accuracy but accepts too many unknown samples. Guard ON with a margin is currently the better balanced demo policy.

## 11. Limitations

- No full-scale MSWC training yet.
- No full EdgeSpot reproduction claim.
- Top500 final test100 artifact is not locked.
- Streaming lacks an official latency/false-alarm benchmark.
- Open-set UI evaluation is sampled demo-level evidence, not a replacement for GSC test100.

## 12. Future Work

- Continue Top500 training when resources are available.
- Produce a complete Top500 test100 artifact.
- Add streaming benchmark metrics.
- Improve calibration and report export.
- Add Playwright UI screenshots and accessibility checks.

## 13. Conclusion

The project selected EdgeSpotFull T4 + SCAF+GE2E based on Microset experiments and used it as the basis for Top500 scaling. The locked Microset result is the current thesis anchor. Top500 epoch13 is a promising local artifact for demo and preliminary analysis. The improved demo UI makes inference decisions, misses, and open-set tradeoffs visible and easier to explain.
