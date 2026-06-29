# Colab MSWC Cap220 FLAC Summary

Run id: `colab_mswc_heavy_flac_target6000000_20260604_171246`

Source log: user-pasted Colab run log, checked on 2026-06-05.

## Data

| Item | Value |
|---|---:|
| Metadata total | 6,624,343 clips, 38,174 words |
| Train words | 37,387 |
| Validation words | 763 |
| Skipped words | 24 words with count < 1 |
| Selected cap | max 220 clips/word |
| Train manifest | 2,012,579 FLAC files |
| Validation manifest | 37,138 FLAC files |
| Total manifest | 2,049,717 FLAC files |
| Disk after FLAC conversion | 110G / 236G, 47% |

Note: this run did not reach the intended `TARGET_FILES=6000000`; the selected cap was limited by the configured range `[180, 220]`. Report this as `MSWC cap220 FLAC`, not as a 6M-clip run.

## GSC-Dev Checkpoint Selection

| Pipeline | Best epoch | ACC@1%FAR |
|---|---:|---:|
| DSCNN-L + PCEN + GE2E | 9 | 85.08% |
| EdgeSpotFull T4 + PCEN + GE2E | 9 | 82.87% |

## Final Metrics

| Pipeline | Split | AUC | EER | FRR@5%FAR | ACC@5%FAR | Keyword ACC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | dev30 | 93.85 +/- 0.48 | 12.88 +/- 0.83 | 23.71 +/- 1.76 | 88.05 +/- 0.56 | 88.79 +/- 0.88 | 80.52 +/- 1.16 |
| DSCNN-L + PCEN + GE2E | test100 | 93.87 +/- 0.47 | 12.78 +/- 0.80 | 23.66 +/- 2.23 | 88.23 +/- 0.68 | 89.82 +/- 1.16 | 80.67 +/- 1.12 |
| EdgeSpotFull T4 + PCEN + GE2E | dev30 | 92.30 +/- 0.54 | 15.51 +/- 0.79 | 31.14 +/- 2.35 | 85.56 +/- 0.72 | 87.07 +/- 0.97 | 76.90 +/- 1.07 |
| EdgeSpotFull T4 + PCEN + GE2E | test100 | 91.31 +/- 0.62 | 16.47 +/- 0.78 | 30.75 +/- 2.44 | 86.03 +/- 0.70 | 88.29 +/- 1.06 | 75.61 +/- 1.06 |

## Interpretation

- DSCNN-L + PCEN + GE2E is the stronger accuracy candidate on this larger FLAC capped subset.
- EdgeSpotFull T4 + PCEN + GE2E remains useful as the compact edge/device candidate.
- On GSC-test100, DSCNN-L is better than EdgeSpotFull T4 by:
  - +2.20 percentage points ACC@5%FAR;
  - +2.56 percentage points AUC;
  - -3.69 percentage points EER;
  - +5.06 percentage points F1.
- Compared with previous manifest50 runs, cap220 substantially improves both models, so increasing MSWC coverage helps.

## Claim Hygiene

- Valid claim: `MSWC cap220 FLAC improves the shortlist evidence and supports DSCNN-L + PCEN + GE2E as the highest-accuracy path`.
- Valid claim: `EdgeSpotFull T4 is smaller and remains a deployment-oriented alternative, but it trails DSCNN-L in accuracy`.
- Do not claim: `trained on full 6M clips`, because this run used about 2.05M FLAC clips.
- Do not claim Drive sync is verified unless the Drive folder is checked; the pasted log ends at `Final artifact sync`.
