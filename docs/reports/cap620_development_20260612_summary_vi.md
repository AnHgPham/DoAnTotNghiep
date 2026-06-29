# Bao cao nhanh - Cap620 development run 2026-06-12

Nguon log: attachment `pasted-text.txt`, run id
`colab_mswc_cap620_development_20260612_050614`.

## Cau hinh run

- Runtime: Google Colab A100 40GB.
- Data: MSWC English cap620 FLAC.
- Train words: `37,387`.
- Val words: `763`.
- Train files: `2,989,780`.
- Val files: `52,399`.
- Runner toggles:
  - `RUN_ACCURACY=1`
  - `RUN_COMPACT=1`
  - `RUN_KD=0`
  - `RUN_SCAF_ABLATION=0`
- Checkpoint selection: GSC-dev composite metric =
  mean(`ACC@1%FAR`, `AUC`, `F1`).
- Final eval: `dev30_far1`, `test100_far1`, `test100_far5`.

## Ket qua test100

| Cau hinh | FAR | ACC | AUC | EER | F1 | Keyword ACC | Ghi chu |
|---|---:|---:|---:|---:|---:|---:|---|
| DSCNN-L + PCEN + GE2E, ep300 composite | 1% | `86.36 +/- 1.29` | `95.21 +/- 0.45` | `11.32 +/- 0.78` | `82.73 +/- 1.11` | `92.92 +/- 0.87` | Best overall moi |
| DSCNN-L + PCEN + GE2E, ep300 composite | 5% | `89.93 +/- 0.65` | `95.21 +/- 0.45` | `11.32 +/- 0.78` | `82.73 +/- 1.11` | `92.92 +/- 0.87` | FAR5 rat manh |
| EdgeSpotFull T4 + PCEN + Triplet hard, ep300 composite | 1% | `69.10 +/- 0.15` | `53.40 +/- 0.48` | `47.84 +/- 0.62` | `39.99 +/- 0.60` | `16.36 +/- 1.46` | Hard-triplet collapse |
| EdgeSpotFull T4 + PCEN + Triplet hard, ep300 composite | 5% | `66.89 +/- 0.23` | `53.40 +/- 0.48` | `47.84 +/- 0.62` | `39.99 +/- 0.60` | `16.36 +/- 1.46` | Khong dung lam ket qua chinh |
| EdgeSpotFull T4 + PCEN + GE2E, ep300 composite | 1% | `82.87 +/- 1.22` | `92.41 +/- 0.44` | `14.82 +/- 0.70` | `77.85 +/- 0.97` | `87.29 +/- 1.19` | Best compact moi |
| EdgeSpotFull T4 + PCEN + GE2E, ep300 composite | 5% | `86.76 +/- 0.59` | `92.41 +/- 0.44` | `14.82 +/- 0.70` | `77.85 +/- 0.97` | `87.29 +/- 1.19` | Compact FAR5 tot |

## So sanh voi fixed 16-pipeline cap620

Baseline fixed 16-pipeline:

- `DSCNN-L + PCEN + GE2E`: `ACC@1%FAR=82.34 +/- 1.19`,
  `AUC=92.42`, `EER=14.89`, `F1=77.75`.
- `EdgeSpotFull T4 + PCEN + GE2E`: `ACC@1%FAR=79.98 +/- 0.98`,
  `AUC=87.23`, `EER=20.23`, `F1=70.68`.
- `EdgeSpotFull T4 + PCEN + Triplet`: `ACC@1%FAR=79.58 +/- 1.35`,
  `AUC=89.85`, `EER=18.22`, `F1=73.29`.

Development run improvement:

- `DSCNN-L + PCEN + GE2E` tang tu `82.34` len `86.36`
  (`+4.02` diem ACC@1%FAR), AUC tang tu `92.42` len `95.21`,
  EER giam tu `14.89` xuong `11.32`, F1 tang tu `77.75` len `82.73`.
- `EdgeSpotFull T4 + PCEN + GE2E` tang tu `79.98` len `82.87`
  (`+2.89` diem ACC@1%FAR), AUC tang tu `87.23` len `92.41`,
  EER giam tu `20.23` xuong `14.82`, F1 tang tu `70.68` len `77.85`.
- `EdgeSpotFull T4 + PCEN + Triplet hard` bi collapse so voi baseline
  Triplet fixed. Nguyen nhan kha nang cao la `--mining hard` qua gay gat
  khi ket hop hard-pair episode seeding tren cap620.

## So sanh voi EdgeSpot-4 paper

Moc paper EdgeSpot-4 duoc dung trong thesis: `82.0% ACC@1%FAR`.

- `DSCNN-L + PCEN + GE2E` dat `86.36 +/- 1.29`, vuot moc paper ve so
  trung binh, nhung day la model DSCNN-L lon hon, khong phai EdgeSpot compact.
- `EdgeSpotFull T4 + PCEN + GE2E` dat `82.87 +/- 1.22`, cao hon `82.0`
  ve so trung binh. Nen viet than trong: "competitive and slightly above the
  EdgeSpot-4 reported mean under our GSC test100 protocol", vi bien do chi
  khoang `+0.87` diem va van trong khoang sai so.
- Chua co KD trong run nay (`RUN_KD=0`), nen khong duoc claim da tai lap
  day du cong thuc EdgeSpot paper co knowledge distillation.

## Khuyen nghi tiep theo

1. Lay `DSCNN-L + PCEN + GE2E, ep300 composite` lam ket qua accuracy chinh.
2. Lay `EdgeSpotFull T4 + PCEN + GE2E, ep300 composite` lam ket qua compact
   chinh hien tai.
3. Khong dung `Triplet hard` lam bang chung chinh. Neu muon cuu Triplet, chay
   ablation rieng voi `semi_hard`, giam `hard_pair_prob` ve `0.10-0.20`, hoac
   tat hard-pair seeding.
4. Neu muc tieu paper-level EdgeSpot compact manh hon, chay nhanh KD rieng
   sau, nhung phai so sanh KD voi baseline cung subset/full profile.
