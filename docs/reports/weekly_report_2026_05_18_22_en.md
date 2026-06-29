# Weekly Report 2026-05-18 to 2026-05-22

## 1. Overview

This week focused on four areas:

1. Microset experiments and model selection.
2. Top500 training and artifact handling.
3. Demo/UI improvements for detection, long audio, and open-set rejection.
4. Documentation and reporting structure.

The main technical conclusion is that Microset experiments selected **EdgeSpotFull T4 + SCAF+GE2E** as the best current direction among the tested configurations. This configuration was then used for Top500 scaling.

## 2. Microset Experiments

Microset was used as the main controlled experiment because it has official CSV splits and is small enough for repeated model comparisons. The tested directions included:

- DSCNN-L + MFCC + Triplet baseline;
- EdgeSpotFull T4 + mel-PCEN + SCAF;
- EdgeSpotFull T4 + mel-PCEN + SCAF+GE2E.

The selected configuration is:

- model: EdgeSpotFull T4;
- feature: mel-PCEN;
- loss: SCAF+GE2E;
- checkpoint: epoch05;
- GSC test100:
  - ACC@5%FAR: 86.12%;
  - KW-ACC: 77.66%;
  - F1: 82.41%;
  - AUC: 95.61%;
  - EER: 11.54%.

This result is the current thesis anchor.

## 3. Top500 Training

Top500 was started after Microset model selection to increase word and speaker diversity.

The first Top500 run showed promising logs, but complete local artifacts were not downloaded/packaged before the session was lost. Therefore it should only be mentioned as progress history if no complete artifact is available.

The second Top500 run used safer artifact handling:

- save every epoch;
- save latest every epoch;
- save to Drive;
- use session-first dataset setup.

This run stopped at epoch13 because of Colab/session/unit limits. The reliable local artifact is:

```text
server/final_kws_artifacts_package/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt
```

Epoch13 dev30 metrics:

- ACC@1%FAR: 86.68%;
- ACC@5%FAR: 88.87%;
- FRR@5%FAR: 20.36%;
- AUC: 95.12%;
- F1: 81.71%.

Top500 is promising but currently preliminary.

## 4. Demo/UI Improvements

The demo now includes:

- model switcher for Microset, Top500, and legacy model when available;
- enrollment presets;
- single detection with top-3/L2/threshold/margin;
- long-audio timeline and miss explanation;
- open-set 17 known / 17 unknown sampled evaluation;
- calibration with balanced score;
- React/Vite UI scaffold.

Current open-set demo recommendation:

- Guard ON;
- Per-class OFF;
- accept margin 0.05.

Guard OFF improves keyword recognition but accepts too many unknown samples.

## 5. Problems

- Colab session resets and unit limits interrupted Top500.
- Copying large WAV caches to Drive was too slow.
- Open-set thresholding has a real tradeoff between false accept and false reject.
- Long-audio segmentation can mismatch label count and detection count.

## 6. Current Artifact Story

| Artifact | Status | Use |
|---|---|---|
| Microset EdgeSpotFull T4 + SCAF+GE2E epoch05 | official locked | thesis main |
| Top500 EdgeSpotFull T4 + SCAF+GE2E epoch13 | local available | demo/preliminary |
| Top500 epoch25 historical | progress only | do not use as final claim unless artifact is recovered |

## 7. Next Work

1. Finish React UI polish and screenshot verification.
2. Finish technical documentation and thesis draft.
3. Resume/rerun Top500 when resources are available.
4. Add streaming benchmark metrics.
5. Standardize export reports.
