# Bao cao so sanh file thesis Word

Ngay doc: 2026-06-04

File nguoi dung:

- Ban dang viet: `D:\Downloads\Đồ Án.docx`
- Ban tham khao: `D:\Downloads\Đồ án (1).docx`

## Ket qua doc file

| File | Vai tro | Kich thuoc | Paragraphs | Tables | Nhan xet nhanh |
|---|---|---:|---:|---:|---|
| `Đồ Án.docx` | Ban cua ban | 7,589 bytes | 49 | 0 | Moi co acknowledgements va outline; hau het cac muc chua co noi dung. |
| `Đồ án (1).docx` | File tham khao | 12,616,583 bytes | 229 | 2 | Co du cau truc, noi dung sau moi heading, hinh/bang/ket qua. |

## Cau truc ban cua ban hien tai

Ban cua ban dang co cac muc:

- Acknowledgements
- I. Introduction
  - 1.1 Abstract
  - 1.2 Context and motivation
  - 1.3 Project objective
  - 1.4 Desired Outcomes
- II. Dataset
  - 2.1 GSC v2 dataset
  - 2.2 MSWC dataset
  - 2.3 Demand dataset
  - 2.4 Data preprocessing and augmentation
- III Model Architecture
  - 3.1 DSCNN-L
  - 3.2 Edgespott4
  - 3.3 MFCC
  - 3.4 PCEN
  - 3.5 Triplet
  - 3.6 SCAF
  - 3.7 GE2E
  - training
- IV System pipeline
  - 12 pipeline combinations
- V Result
  - 5.1 DSCNN
  - 5.1.1 Confusion Matrix
  - 5.1.2 mAP
  - 5.2 EdgespotFull T4
- VII Conclusion
  - 7.1 Conclusion
  - 7.2 Limitation
  - 7.3 Future work

## Cau truc file tham khao

File tham khao co cach viet ro rang hon:

- Acknowledgments
- Introduction
  - Abstract
  - Context and Motivation
  - Project Objective
  - Desired Outcomes
- Dataset
  - Tung dataset co doan giai thich cu the
  - Preprocessing va Augmentation co bullet chi tiet
- Model Architecture
  - Giai thich tung model/module
  - Co ly do chon model
- System Pipeline
  - Tach module pipeline theo chuc nang
  - Moi module co input/output va vai tro
- Results
  - Tach result theo model/module
  - Co curve, confusion matrix, loss, threshold sweep
- Inference / End-to-end result
- Conclusion / Limitations / Future Work

## Nhung diem can sua ngay trong ban cua ban

1. `Edgespott4` -> `EdgeSpotFull T4`.
2. `Demand dataset` -> `DEMAND Noise Dataset`.
3. `training` -> nen doi thanh `3.8 Training Objective and Loss Design`.
4. `4.11EdgeSpotFull...` -> thieu dau cach: `4.11 EdgeSpotFull T4 + PCEN + GE2E`.
5. `V Result` nen doi thanh `V. Experimental Results and Discussion`.
6. `Confusion Matrix` va `mAP` khong phai metric chinh cua KWS open-set. Nen thay bang:
   - `ACC@1%FAR`
   - `ACC@5%FAR`
   - `FRR@5%FAR`
   - `AUC`
   - `EER`
   - `F1`
   - `DET Curve`
7. Nen them muc `VI. Demo System` hoac `VI. Inference and Demo System` neu thesis co trinh bay web demo.
8. Neu theo format hoc thuat hon, nen de `Abstract` rieng truoc Chapter 1. Neu muon bam sat file tham khao thi co the giu `1.1 Abstract`, nhung phai nhat quan.

## Noi dung can chen vao tung muc

### Acknowledgements

Ban hien tai co y dung, nhung can sua ngu phap:

- `support and appreciation for my supportive` bi lap tu.
- `encourgarement` sai chinh ta, dung `encouragement`.
- `supportive emotion` khong tu nhien, dung `emotional support`.
- `wonderfull time to reading this` sai, dung `a wonderful time reading this`.

Nen dung ban gon:

```text
First and foremost, I would like to express my sincere gratitude to my supervisor, Dr. Tran Hoang Tung, for his guidance, insightful feedback, and continuous support throughout my internship.

I would also like to thank Dr. Tran Giang Son for providing access to the ICTLab server, which made the computational experiments in this project possible.

I am deeply grateful to my family for their encouragement and emotional support, which helped me stay motivated during the project.

Finally, I would like to thank my friends for their support in reviewing and improving this thesis.
```

### I. Introduction

Can viet 4 y:

1. KWS la gi va ung dung o dau.
2. Vi sao closed-set KWS chua du cho keyword ca nhan hoa.
3. Vi sao few-shot va open-set rejection quan trong.
4. Do an nay tap trung vao pipeline embedding/prototype va so sanh model-feature-loss.

### II. Dataset

Can co noi dung cho:

- GSC v2: dung de evaluate few-shot open-set, khong phai train chinh.
- MSWC: dung train Microset/Top500/Full MSWC.
- DEMAND: dung noise augmentation.
- Preprocessing: resample 16 kHz, mono, trim/pad 1s, feature extraction.
- Augmentation: noise, time shift, SpecAugment.

### III. Model Architecture

Khong nen de MFCC/PCEN/Triplet/SCAF/GE2E ngang hang voi model architecture hoan toan. Nen viet thanh:

- `3.1 DSCNN-L Baseline`
- `3.2 EdgeSpotFull T4 Encoder`
- `3.3 Audio Frontends: MFCC and PCEN`
- `3.4 Metric Learning Objectives`
  - Triplet
  - SCAF
  - GE2E
  - SCAF+GE2E

### IV. System Pipeline

Phan 12/16 pipeline nen viet thanh experiment design, khong nen chi liet ke. Can giai thich:

- Architecture axis: DSCNN-L vs EdgeSpotFull T4.
- Frontend axis: MFCC vs PCEN.
- Loss axis: Triplet, SCAF, GE2E, SCAF+GE2E.
- Muc dich: ablation de biet thanh phan nao co tac dong manh nhat.

### V. Results

Nen chia theo muc do evidence:

1. Microset: evidence chinh de chon huong EdgeSpotFull T4 + PCEN + SCAF+GE2E.
2. Full MSWC 16-pipeline phase-1: ablation/screening, khong phai final.
3. Full MSWC shortlist manifest20/50: evidence chon DSCNN-L + PCEN + GE2E la accuracy candidate, EdgeSpotFull T4 + PCEN + GE2E la compact candidate.
4. Top500 epoch13: reproducible artifact/preliminary evidence.

## Claim nen dung

- Microset supports EdgeSpotFull T4 + PCEN + SCAF+GE2E as the main compact architecture direction.
- Full MSWC shortlist shows DSCNN-L + PCEN + GE2E currently achieves higher accuracy.
- EdgeSpotFull T4 remains valuable because it has far fewer parameters.
- PCEN and GE2E are the most consistent positive components in Full MSWC ablation.

## Claim khong nen dung

- Khong viet `EdgeSpot is always better than DSCNN`.
- Khong viet `SCAF+GE2E is always the best`.
- Khong viet `This fully reproduces EdgeSpot paper`.
- Khong dung UI sampled open-set demo thay cho `gsc_edgespot_exact test100`.
- Khong goi Top500 epoch25 la final artifact neu chua co checkpoint/result JSON.

## Buoc tiep theo de hoan thanh thesis

1. Sua Acknowledgements trong `Đồ Án.docx`.
2. Dien noi dung Introduction tu file `docs/thesis/thesis_intro_vi_guidance_2026_06_04.md`.
3. Viet Dataset va Methodology theo KWS, khong copy logic ADAS tu file tham khao.
4. Tao bang result chuan tu `reports/microset/result_table.md`, `reports/full_mswc_matrix_analysis/matrix_best_epoch_metrics.md`, va shortlist report.
5. Chen hinh DET curve, heatmap, ranked bar tu `reports/full_mswc_matrix_analysis/`.
6. Viet Conclusion theo huong: accuracy candidate vs compact candidate.
