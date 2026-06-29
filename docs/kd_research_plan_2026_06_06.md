# Ke Hoach Nghien Cuu KD Cho KWS

Ngay luu: 2026-06-06

## Summary

Sau khi kiem tra lai EdgeSpot paper, can sua lai cach nhin ve KD: KD trong paper khong phai random teacher. Paper dung pretrained Wav2Vec2.0 teacher, them head giam chieu ve 64-D, head nay duoc toi uu bang Sub-center ArcFace, roi student EdgeSpot hoc tu teacher embedding bang KD.

Vi vay, KD la mot huong nghiem tuc neu lam dung. Nhung voi project hien tai, neu chi dung `Wav2Vec2Teacher` ma khong co `head_checkpoint`, projection head trong code la random, nen chua duoc claim la KD paper-grade.

Nguon doi chieu: EdgeSpot arXiv 2601.16316, phan Training Methodology.

## Tinh Lai Vi Tri Cua KD

| Huong | Trang thai | Co duoc claim chinh khong? | Nhan xet |
|---|---|---:|---|
| Non-KD hien tai: PCEN + GE2E | Da co ket qua manh | Co | Day van la main thesis evidence hien tai. |
| KD voi random projection head | Code chay duoc | Khong | Chi smoke test, khong co gia tri ket luan khoa hoc. |
| KD voi trained Wav2Vec2 teacher head | Chua co du trong result chinh | Co the | Day moi la KD dung nghia de tinh lai ranking. |
| Full EdgeSpot reproduction | Chua dat | Chua | Can data/training setup gan paper hon: teacher trained, KD+SCAF, train dai hon. |

## So Sanh Ket Qua Hien Co

| So sanh hien tai | ACC@1%FAR | ACC@5%FAR / Open-set ACC | AUC | EER | F1 | Ket luan |
|---|---:|---:|---:|---:|---:|---|
| Full MSWC manifest20: DSCNN-L + PCEN + GE2E | 82.10 | 86.05 | 91.57 | 16.25 | 75.90 | Accuracy-oriented best hien tai |
| Full MSWC manifest20: EdgeSpotFull T4 + PCEN + GE2E | 79.58 | 83.06 | 87.22 | 20.40 | 70.46 | Compact nhung kem DSCNN |
| Chenh lech EdgeSpot so voi DSCNN | -2.52 pp | -2.99 pp | -4.35 pp | +4.15 pp | -5.44 pp | KD dang thu de thu hep gap |
| MSWC cap220 FLAC: DSCNN-L + PCEN + GE2E | - | 88.23 | 93.87 | 12.78 | 80.67 | Manh hon khi train lon hon |
| MSWC cap220 FLAC: EdgeSpotFull T4 + PCEN + GE2E | - | 86.03 | 91.31 | 16.47 | 75.61 | Gap van con |
| Chenh lech EdgeSpot cap220 so voi DSCNN | - | -2.20 pp | -2.56 pp | +3.69 pp | -5.06 pp | KD co muc tieu ro rang |

Diem quan trong: KD dang thu nhat cho EdgeSpot, khong phai DSCNN. Vi KD trong paper duoc thiet ke de giup student nho hoc embedding tu teacher lon.

## Chien Luoc KD Nghiem Tuc

Baseline bat buoc giu nguyen:

- `EdgeSpotFull T4 + PCEN + GE2E`
- `DSCNN-L + PCEN + GE2E`

KD candidate nen chay theo thu tu:

1. `EdgeSpotFull T4 + PCEN + kd_scaf`
   - Gan paper nhat: KD + SCAF.
2. `EdgeSpotFull T4 + PCEN + kd_ge2e`
   - Bien the theo huong project, vi GE2E dang hop voi prototype evaluation.
3. `EdgeSpotFull T4 + PCEN + kd_scaf_ge2e`
   - Chi chay sau khi 2 huong tren on; can tune weight de tranh SCAF pha GE2E.

Teacher setup bat buoc:

- Dung Wav2Vec2.0 pretrained.
- Train hoac co checkpoint cho projection/dimensionality-reduction head 64-D.
- Sau do moi precompute teacher embeddings.
- Khong dung random head cho result chinh.

Loss nen tinh nhu sau:

```text
Paper-aligned:
L = L_KD + 5e-5 * L_SCAF

Project GE2E variant:
L = L_GE2E + lambda_kd * L_KD

Full hybrid:
L = L_GE2E + lambda_kd * L_KD + 5e-5 * L_SCAF
```

## Decision Rules

- Neu KD giup EdgeSpot tang it nhat:
  - `+1.0 pp ACC@1%FAR` tren GSC-test100,
  - AUC khong giam,
  - EER giam hoac khong xau hon trong sai so,
  - F1 khong giam,
  thi KD duoc dua vao thesis nhu mot ablation quan trong.

- Neu KD giup EdgeSpot vuot DSCNN-L:
  - Cap nhat ket luan chinh thanh:
    ```text
    EdgeSpotFull T4 + PCEN + KD-based loss is the best accuracy/compactness trade-off.
    ```
  - DSCNN-L tro thanh accuracy baseline, khong con la candidate chinh.

- Neu KD chi giup EdgeSpot nhung chua vuot DSCNN:
  - Ket luan:
    ```text
    KD improves the compact EdgeSpot student, but DSCNN-L + PCEN + GE2E remains the highest-accuracy model.
    ```

- Neu KD khong cai thien:
  - Giu ket luan hien tai:
    ```text
    Direct GE2E training is sufficient for the current thesis setting; KD remains future work.
    ```

## Test Plan

Stage 1: Teacher validation

- Kiem tra teacher embedding dim = 64.
- Kiem tra teacher head khong random.
- Kiem tra teacher embedding co du path cho train manifest.

Stage 2: Small KD ablation

- Data: Top500Full hoac MSWC cap nho.
- Chay:
  - `EdgeSpotFull T4 + PCEN + GE2E`
  - `EdgeSpotFull T4 + PCEN + kd_scaf`
  - `EdgeSpotFull T4 + PCEN + kd_ge2e`
- Eval: GSC-dev30 va GSC-test100.

Stage 3: Final KD run neu Stage 2 tot

- Data: cung setting voi current strongest run.
- Eval bat buoc:
  - `ACC@1%FAR`
  - `ACC@5%FAR`
  - `AUC`
  - `EER`
  - `FRR@5%FAR`
  - `F1`
  - DET curve
- So sanh cung bang voi non-KD.

## Assumptions

- Hien tai project chua co trained teacher projection head dang claim.
- KD chi duoc tinh lai nghien cuu neu teacher hop le, khong phai random head.
- Muc tieu thuc te cua KD la nang EdgeSpotFull T4, vi day la model nho cho edge/device.
- Ket luan thesis chi thay doi theo GSC-test100, khong thay doi theo train loss hoac dev-only result.

