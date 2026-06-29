| effect                | scope                       | delta_acc1_pp | delta_acc5_pp | delta_frr5_pp | delta_f1_pp | note                                                            |
| --------------------- | --------------------------- | ------------- | ------------- | ------------- | ----------- | --------------------------------------------------------------- |
| PCEN vs MFCC          | DSCNN-L + Triplet           | +2.94 pp      | +5.36 pp      | -16.97 pp     | +11.80 pp   | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | DSCNN-L + SCAF              | -0.70 pp      | -0.59 pp      | +0.79 pp      | -0.27 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | DSCNN-L + GE2E              | +4.37 pp      | +6.20 pp      | -17.27 pp     | +6.99 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | DSCNN-L + SCAF+GE2E         | +0.87 pp      | +1.02 pp      | -1.88 pp      | -1.29 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | EdgeSpotFull T4 + Triplet   | +1.76 pp      | +2.46 pp      | -8.37 pp      | +7.15 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | EdgeSpotFull T4 + SCAF      | +0.02 pp      | -0.30 pp      | +1.52 pp      | -0.44 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | EdgeSpotFull T4 + GE2E      | +3.63 pp      | +6.00 pp      | -20.49 pp     | +17.91 pp   | Tác động của frontend khi giữ nguyên model và loss.             |
| PCEN vs MFCC          | EdgeSpotFull T4 + SCAF+GE2E | +0.09 pp      | +0.06 pp      | -2.97 pp      | +0.26 pp    | Tác động của frontend khi giữ nguyên model và loss.             |
| GE2E vs Triplet       | DSCNN-L + MFCC              | +3.00 pp      | +6.47 pp      | -25.28 pp     | +17.51 pp   | Tác động của loss centroid/prototype so với loss pairwise.      |
| SCAF+GE2E vs SCAF     | DSCNN-L + MFCC              | -1.48 pp      | -1.43 pp      | +1.82 pp      | +0.15 pp    | Tác động khi thêm GE2E vào SCAF với trọng số hiện tại.          |
| SCAF+GE2E vs GE2E     | DSCNN-L + MFCC              | -3.06 pp      | -6.95 pp      | +25.94 pp     | -21.53 pp   | Kiểm tra liệu hybrid hiện tại có tốt hơn GE2E đơn lẻ hay không. |
| GE2E vs Triplet       | DSCNN-L + PCEN              | +4.43 pp      | +7.31 pp      | -25.58 pp     | +12.70 pp   | Tác động của loss centroid/prototype so với loss pairwise.      |
| SCAF+GE2E vs SCAF     | DSCNN-L + PCEN              | +0.09 pp      | +0.18 pp      | -0.85 pp      | -0.87 pp    | Tác động khi thêm GE2E vào SCAF với trọng số hiện tại.          |
| SCAF+GE2E vs GE2E     | DSCNN-L + PCEN              | -6.56 pp      | -12.13 pp     | +41.33 pp     | -29.81 pp   | Kiểm tra liệu hybrid hiện tại có tốt hơn GE2E đơn lẻ hay không. |
| GE2E vs Triplet       | EdgeSpotFull T4 + MFCC      | +0.31 pp      | +1.00 pp      | -4.12 pp      | +2.17 pp    | Tác động của loss centroid/prototype so với loss pairwise.      |
| SCAF+GE2E vs SCAF     | EdgeSpotFull T4 + MFCC      | -0.35 pp      | -0.71 pp      | +4.37 pp      | -2.74 pp    | Tác động khi thêm GE2E vào SCAF với trọng số hiện tại.          |
| SCAF+GE2E vs GE2E     | EdgeSpotFull T4 + MFCC      | -0.16 pp      | -0.41 pp      | +3.33 pp      | -1.41 pp    | Kiểm tra liệu hybrid hiện tại có tốt hơn GE2E đơn lẻ hay không. |
| GE2E vs Triplet       | EdgeSpotFull T4 + PCEN      | +2.18 pp      | +4.54 pp      | -16.24 pp     | +12.93 pp   | Tác động của loss centroid/prototype so với loss pairwise.      |
| SCAF+GE2E vs SCAF     | EdgeSpotFull T4 + PCEN      | -0.28 pp      | -0.35 pp      | -0.12 pp      | -2.04 pp    | Tác động khi thêm GE2E vào SCAF với trọng số hiện tại.          |
| SCAF+GE2E vs GE2E     | EdgeSpotFull T4 + PCEN      | -3.70 pp      | -6.35 pp      | +20.85 pp     | -19.07 pp   | Kiểm tra liệu hybrid hiện tại có tốt hơn GE2E đơn lẻ hay không. |
| EdgeSpotFull vs DSCNN | MFCC + Triplet              | -0.30 pp      | -0.96 pp      | +4.12 pp      | -5.98 pp    | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | MFCC + SCAF                 | -1.22 pp      | -0.61 pp      | +0.12 pp      | +1.70 pp    | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | MFCC + GE2E                 | -2.99 pp      | -6.43 pp      | +25.28 pp     | -21.31 pp   | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | MFCC + SCAF+GE2E            | -0.09 pp      | +0.11 pp      | +2.67 pp      | -1.19 pp    | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | PCEN + Triplet              | -1.48 pp      | -3.86 pp      | +12.72 pp     | -10.62 pp   | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | PCEN + SCAF                 | -0.50 pp      | -0.32 pp      | +0.85 pp      | +1.53 pp    | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | PCEN + GE2E                 | -3.73 pp      | -6.63 pp      | +22.06 pp     | -10.39 pp   | Tác động của backbone khi giữ nguyên frontend và loss.          |
| EdgeSpotFull vs DSCNN | PCEN + SCAF+GE2E            | -0.87 pp      | -0.85 pp      | +1.58 pp      | +0.36 pp    | Tác động của backbone khi giữ nguyên frontend và loss.          |