DET summary from best checkpoint per experiment. Lower EER/FRR is better; higher AUC/Recall/ACC is better. Full threshold-by-threshold DET curves require saved raw scores, which are not present in these phase-1 logs.

| rank_eer | combo                              | best_epoch | AUC    | EER    | FRR@1%FAR | FRR@5%FAR | Recall@1%FAR | Recall@5%FAR | ACC@1%FAR | ACC@5%FAR |
| -------- | ---------------------------------- | ---------- | ------ | ------ | --------- | --------- | ------------ | ------------ | --------- | --------- |
| 1        | DSCNN-L + PCEN + GE2E              | 5          | 85.89% | 22.60% | 72.00%    | 48.97%    | 28.00%       | 51.03%       | 76.67%    | 79.98%    |
| 2        | DSCNN-L + MFCC + GE2E              | 5          | 78.59% | 28.37% | 84.85%    | 66.24%    | 15.15%       | 33.76%       | 72.30%    | 73.78%    |
| 3        | EdgeSpotFull T4 + PCEN + GE2E      | 4          | 76.68% | 31.30% | 84.42%    | 71.03%    | 15.58%       | 28.97%       | 72.94%    | 73.35%    |
| 4        | DSCNN-L + PCEN + Triplet           | 5          | 73.71% | 33.38% | 87.15%    | 74.55%    | 12.85%       | 25.45%       | 72.24%    | 72.67%    |
| 5        | EdgeSpotFull T4 + PCEN + Triplet   | 5          | 60.18% | 43.41% | 92.48%    | 87.27%    | 7.52%        | 12.73%       | 70.76%    | 68.81%    |
| 6        | DSCNN-L + MFCC + Triplet           | 5          | 57.93% | 44.57% | 97.39%    | 91.52%    | 2.61%        | 8.48%        | 69.30%    | 67.31%    |
| 7        | EdgeSpotFull T4 + MFCC + SCAF      | 5          | 53.98% | 47.07% | 96.91%    | 90.48%    | 3.09%        | 9.52%        | 69.50%    | 67.65%    |
| 8        | EdgeSpotFull T4 + PCEN + SCAF      | 1          | 53.59% | 47.57% | 97.40%    | 92.00%    | 2.60%        | 8.00%        | 69.52%    | 67.35%    |
| 9        | EdgeSpotFull T4 + MFCC + GE2E      | 5          | 52.66% | 48.49% | 97.82%    | 91.52%    | 2.18%        | 8.48%        | 69.31%    | 67.35%    |
| 10       | DSCNN-L + MFCC + SCAF+GE2E         | 2          | 52.04% | 48.71% | 97.27%    | 92.18%    | 2.73%        | 7.82%        | 69.24%    | 66.83%    |
| 11       | DSCNN-L + MFCC + SCAF              | 1          | 52.03% | 48.85% | 94.06%    | 90.36%    | 5.94%        | 9.64%        | 70.72%    | 68.26%    |
| 12       | DSCNN-L + PCEN + SCAF              | 1          | 52.32% | 49.15% | 95.27%    | 91.15%    | 4.73%        | 8.85%        | 70.02%    | 67.67%    |
| 13       | EdgeSpotFull T4 + PCEN + SCAF+GE2E | 3          | 50.95% | 49.69% | 97.58%    | 91.88%    | 2.42%        | 8.12%        | 69.24%    | 67.00%    |
| 14       | EdgeSpotFull T4 + MFCC + SCAF+GE2E | 5          | 50.52% | 49.99% | 98.12%    | 94.85%    | 1.88%        | 5.15%        | 69.15%    | 66.94%    |
| 15       | DSCNN-L + PCEN + SCAF+GE2E         | 1          | 50.05% | 50.11% | 95.15%    | 90.30%    | 4.85%        | 9.70%        | 70.11%    | 67.85%    |
| 16       | EdgeSpotFull T4 + MFCC + Triplet   | 5          | 48.86% | 50.78% | 99.27%    | 95.64%    | 0.73%        | 4.36%        | 69.00%    | 66.35%    |