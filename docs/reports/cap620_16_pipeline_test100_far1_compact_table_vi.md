# B?ng full 16 pipeline - GSC test100 t?i FAR=1%

C?c gi? tr? l? mean ? std qua 100 runs. Metric gi? l?i: ACC@1%FAR, AUC, EER, FRR@1%FAR v? F1.

| # | Backbone | Frontend | Loss | ACC@1%FAR | AUC | EER | FRR@1%FAR | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DSCNN-L | MFCC | Triplet | 71.52 ? 0.67 | 75.55 ? 1.09 | 31.41 ? 0.98 | 89.67 ? 2.58 | 57.17 ? 1.11 |
| 2 | DSCNN-L | MFCC | SCAF | 70.08 ? 0.44 | 55.04 ? 1.11 | 46.41 ? 0.85 | 94.87 ? 1.70 | 41.38 ? 0.83 |
| 3 | DSCNN-L | MFCC | GE2E | 77.08 ? 1.24 | 86.46 ? 0.83 | 21.95 ? 1.03 | 70.49 ? 4.48 | 68.50 ? 1.30 |
| 4 | DSCNN-L | MFCC | SCAF+GE2E | 69.04 ? 0.41 | 47.78 ? 1.21 | 52.15 ? 1.67 | 98.54 ? 1.48 | 35.93 ? 1.54 |
| 5 | DSCNN-L | PCEN | Triplet | 79.98 ? 1.11 | 90.65 ? 0.62 | 17.57 ? 0.93 | 62.37 ? 3.81 | 74.14 ? 1.23 |
| 6 | DSCNN-L | PCEN | SCAF | 69.44 ? 0.00 | 50.00 ? 0.00 | 50.00 ? 0.00 | 100.00 ? 0.00 | 0.00 ? 0.00 |
| 7 | DSCNN-L | PCEN | GE2E | 82.34 ? 1.19 | 92.42 ? 0.54 | 14.89 ? 0.84 | 54.55 ? 4.01 | 77.75 ? 1.15 |
| 8 | DSCNN-L | PCEN | SCAF+GE2E | 69.44 ? 0.00 | 50.00 ? 0.00 | 50.00 ? 0.00 | 100.00 ? 0.00 | 0.00 ? 0.00 |
| 9 | EdgeSpotFull T4 | MFCC | Triplet | 69.63 ? 0.34 | 52.84 ? 0.95 | 48.05 ? 1.02 | 96.92 ? 1.22 | 39.79 ? 0.98 |
| 10 | EdgeSpotFull T4 | MFCC | SCAF | 69.44 ? 0.00 | 50.00 ? 0.00 | 50.00 ? 0.00 | 100.00 ? 0.00 | 0.00 ? 0.00 |
| 11 | EdgeSpotFull T4 | MFCC | GE2E | 70.76 ? 0.39 | 65.30 ? 1.12 | 39.05 ? 1.07 | 92.29 ? 1.43 | 48.82 ? 1.12 |
| 12 | EdgeSpotFull T4 | MFCC | SCAF+GE2E | 69.67 ? 1.01 | 50.88 ? 0.90 | 50.39 ? 0.82 | 96.10 ? 3.63 | 37.57 ? 0.76 |
| 13 | EdgeSpotFull T4 | PCEN | Triplet | 79.58 ? 1.35 | 89.85 ? 0.63 | 18.22 ? 0.78 | 62.21 ? 4.82 | 73.29 ? 1.02 |
| 14 | EdgeSpotFull T4 | PCEN | SCAF | 69.44 ? 0.00 | 50.00 ? 0.00 | 50.00 ? 0.00 | 100.00 ? 0.00 | 0.00 ? 0.00 |
| 15 | EdgeSpotFull T4 | PCEN | GE2E | 79.98 ? 0.98 | 87.23 ? 0.75 | 20.23 ? 0.96 | 61.26 ? 3.39 | 70.68 ? 1.23 |
| 16 | EdgeSpotFull T4 | PCEN | SCAF+GE2E | 69.44 ? 0.00 | 50.00 ? 0.00 | 50.00 ? 0.00 | 100.00 ? 0.00 | 0.00 ? 0.00 |

Ghi ch?: c?c d?ng SCAF/SCAF+GE2E c? ACC quanh 69% nh?ng F1 b?ng 0 ho?c r?t th?p l? d?u hi?u reject/collapse, kh?ng ???c xem l? t?t.
