# Server FAR Metrics Summary

Source: JSON files copied from ict6 `/storage/<user>/an_kws/DoAnTotNghiep/results`.

| Dataset/Run | Pipeline | Split | ACC@1%FAR | ACC@5%FAR | FRR@1%FAR | FRR@5%FAR | AUC | EER | F1 | Keyword ACC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full MSWC manifest20 | DSCNN-L + PCEN + GE2E | dev30 | 82.69 +/- 1.26 | 85.98 +/- 0.79 | 52.26 +/- 4.21 | 30.67 +/- 2.43 | 91.96 +/- 0.68 | 15.89 +/- 1.03 | 76.40 +/- 1.39 | 87.96 +/- 1.13 |
| Full MSWC manifest20 | DSCNN-L + PCEN + GE2E | test100 | 82.10 +/- 0.87 | 86.05 +/- 0.66 | 55.48 +/- 2.94 | 31.38 +/- 2.28 | 91.57 +/- 0.58 | 16.25 +/- 0.86 | 75.90 +/- 1.16 | 88.90 +/- 1.25 |
| Full MSWC manifest20 | EdgeSpotFull T4 + PCEN + GE2E | dev30 | 79.35 +/- 1.00 | 82.60 +/- 0.63 | 62.75 +/- 3.49 | 40.07 +/- 2.19 | 88.55 +/- 0.73 | 18.90 +/- 1.03 | 72.40 +/- 1.34 | 82.75 +/- 1.21 |
| Full MSWC manifest20 | EdgeSpotFull T4 + PCEN + GE2E | test100 | 79.58 +/- 0.91 | 83.06 +/- 0.82 | 63.45 +/- 3.18 | 40.01 +/- 2.96 | 87.22 +/- 0.75 | 20.40 +/- 1.01 | 70.46 +/- 1.30 | 83.01 +/- 1.49 |
| Full MSWC manifest50 | DSCNN-L + PCEN + GE2E | dev30 | 81.91 +/- 0.92 | 85.17 +/- 0.87 | 54.50 +/- 3.18 | 32.49 +/- 3.16 | 91.47 +/- 0.67 | 15.88 +/- 1.07 | 76.41 +/- 1.44 | 86.27 +/- 1.12 |
| Full MSWC manifest50 | DSCNN-L + PCEN + GE2E | test100 | 80.96 +/- 1.16 | 84.68 +/- 0.70 | 59.25 +/- 3.99 | 35.95 +/- 2.46 | 90.45 +/- 0.68 | 17.42 +/- 1.08 | 74.34 +/- 1.43 | 87.63 +/- 1.25 |
| Full MSWC manifest50 | EdgeSpotFull T4 + PCEN + GE2E | dev30 | 77.33 +/- 1.04 | 82.36 +/- 0.64 | 69.77 +/- 3.68 | 41.19 +/- 2.10 | 89.11 +/- 0.51 | 18.97 +/- 0.84 | 72.30 +/- 1.09 | 84.30 +/- 1.22 |
| Full MSWC manifest50 | EdgeSpotFull T4 + PCEN + GE2E | test100 | 77.14 +/- 0.89 | 82.24 +/- 0.74 | 71.20 +/- 3.12 | 42.02 +/- 2.53 | 87.74 +/- 0.66 | 20.19 +/- 0.90 | 70.73 +/- 1.16 | 83.49 +/- 1.22 |
| Top500Full recheck | DSCNN-L + PCEN + GE2E | dev30 | 83.96 +/- 1.05 | 87.00 +/- 0.67 | 47.08 +/- 3.85 | 25.70 +/- 2.28 | 94.03 +/- 0.55 | 13.44 +/- 1.01 | 79.74 +/- 1.39 | 87.95 +/- 0.88 |
| Top500Full recheck | DSCNN-L + PCEN + GE2E | test100 | 81.55 +/- 1.17 | 86.56 +/- 0.71 | 57.19 +/- 4.00 | 28.64 +/- 2.08 | 93.17 +/- 0.50 | 14.00 +/- 0.85 | 78.97 +/- 1.18 | 88.62 +/- 1.23 |
| Top500Full epoch13 re-eval | EdgeSpotFull T4 + PCEN + SCAF+GE2E | dev30 | 86.68 +/- 0.81 | 88.88 +/- 0.53 | 38.80 +/- 2.77 | 20.36 +/- 1.67 | 95.12 +/- 0.36 | 12.03 +/- 0.53 | 81.71 +/- 0.74 | 88.86 +/- 1.01 |
| Top500Full epoch13 re-eval | EdgeSpotFull T4 + PCEN + SCAF+GE2E | test100 | 85.62 +/- 1.04 | 88.79 +/- 0.66 | 43.10 +/- 3.56 | 21.59 +/- 2.14 | 95.34 +/- 0.40 | 11.51 +/- 0.76 | 82.45 +/- 1.08 | 90.44 +/- 0.97 |
| Top500Full recheck | EdgeSpotFull T4 + PCEN + SCAF+GE2E | dev30 | 85.20 +/- 1.40 | 87.48 +/- 0.56 | 44.28 +/- 4.92 | 24.84 +/- 1.90 | 93.99 +/- 0.26 | 13.35 +/- 0.50 | 79.87 +/- 0.69 | 85.86 +/- 1.04 |
| Top500Full recheck | EdgeSpotFull T4 + PCEN + SCAF+GE2E | test100 | 83.50 +/- 1.40 | 86.18 +/- 0.66 | 50.03 +/- 4.92 | 29.89 +/- 2.50 | 92.73 +/- 0.47 | 15.11 +/- 0.64 | 77.45 +/- 0.87 | 86.15 +/- 0.93 |

Notes:
- `ACC@1%FAR` is the stricter operating point and should be used when false accepts must be minimized.
- `ACC@5%FAR` is more permissive and is the value printed as `Open-set ACC` in logs when `target_far=0.05`.
- Lower `FRR` and `EER` are better; higher `ACC`, `AUC`, and `F1` are better.
