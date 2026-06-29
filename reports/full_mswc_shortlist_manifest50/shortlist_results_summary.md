# Full MSWC Shortlist Manifest50 Summary

Verified on ict6 and copied locally on 2026-06-02.

## Setup

- Dataset: Full MSWC English extracted clips.
- Manifest cap: max 50 clips per word.
- Train manifest: 939,108 files.
- Validation manifest: 18,598 files.
- Train words: 37,387.
- Validation words: 763.
- Training budget: 20 epochs, 200 episodes/epoch, 30 classes x 10 samples.
- Final evaluation: GSC EdgeSpot-exact protocol, k=10, test100, DET curve enabled.

## Test100 Results

| Pipeline | Params | Runs | ACC@1%FAR | ACC@5%FAR | AUC | EER | FRR@1%FAR | FRR@5%FAR | Keyword ACC | F1 | Best dev selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | 412,900 | 100 | 80.96 +/- 1.16 | 84.68 +/- 0.70 | 90.45 +/- 0.68 | 17.42 +/- 1.08 | 59.25 +/- 3.99 | 35.95 +/- 2.46 | 87.63 +/- 1.25 | 74.34 +/- 1.43 | 82.14 |
| EdgeSpotFull T4 + PCEN + GE2E | 130,594 | 100 | 77.14 +/- 0.89 | 82.24 +/- 0.74 | 87.74 +/- 0.66 | 20.19 +/- 0.90 | 71.20 +/- 3.12 | 42.02 +/- 2.53 | 83.49 +/- 1.22 | 70.73 +/- 1.16 | 77.27 |

## Interpretation

- DSCNN-L remains the higher-accuracy candidate on max50:
  - +3.82 pp ACC@1%FAR over EdgeSpotFull T4.
  - +2.44 pp ACC@5%FAR.
  - +2.71 pp AUC.
  - -2.77 pp EER.
  - +3.61 pp F1.
- EdgeSpotFull T4 remains the compact edge/device candidate:
  - 130,594 params vs 412,900 params for DSCNN-L.
  - About 31.6% of DSCNN-L parameter count.
- Compared with manifest20, both max50 test100 scores are lower:
  - DSCNN-L ACC@1%FAR: 82.10 -> 80.96.
  - EdgeSpotFull T4 ACC@1%FAR: 79.58 -> 77.14.
- This suggests the max50 manifest adds more acoustic/word variability and likely needs a longer or better-tuned training schedule. Do not claim max50 improved final accuracy; claim it is a harder robustness follow-up that confirms the same ranking: DSCNN-L > EdgeSpotFull T4 for accuracy, EdgeSpotFull T4 better for compact deployment.

## Artifacts

- Main TSV: `reports/full_mswc_shortlist_manifest50/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv`
- DSCNN recovery TSV: `reports/full_mswc_shortlist_manifest50/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.tsv`
- Raw JSON/DET artifacts: `reports/full_mswc_shortlist_manifest50/raw/full_mswc_shortlist_manifest50_clips_e20_ep200/`
- Server main log copied locally: `reports/full_mswc_shortlist_manifest50/logs/full_mswc_shortlist_manifest50_clips_e20_ep200.log`
- Server DSCNN recovery log copied locally: `reports/full_mswc_shortlist_manifest50/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.log`

## Notes

- DSCNN initially failed during epoch 14 due an unreadable `.opus`/NFS audio load error.
- `src/data/mswc_dataset.py` was patched to retry same-label samples before falling back to silence.
- After recovery, DSCNN improved best GSC-dev ACC@1%FAR from 78.50 to 82.14 and completed dev30/test100 successfully.
