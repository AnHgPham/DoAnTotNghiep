# Draft noi dung thesis tieng Viet va goi y viet tieng Anh

Ngay soan: 2026-06-04

Pham vi: ban nay dung de dua vao thesis/Google Docs. Noi dung duoc viet bang tieng Viet de nguoi viet tu dich sang tieng Anh, kem cac mau cau va loi ngu phap can tranh.

Nguon da doi chieu:

- `docs/references/M1-Phan_Thanh_Binh-KWS_Master.pdf`: cau truc thesis mau gom Declaration, Acknowledgements, Abstract, Introduction, Background, Methodology, Results & Discussion, Conclusions & Future Works.
- `reports/microset/result_table.md`: bang ket qua Microset chinh.
- `reports/full_mswc_matrix_analysis/matrix_best_epoch_metrics.md`: bang 16 pipeline Full MSWC phase-1.
- `reports/full_mswc_shortlist_manifest20/shortlist_results_summary.md`: ket qua shortlist Full MSWC manifest20.
- `reports/full_mswc_shortlist_manifest50/shortlist_results_summary.md`: ket qua shortlist Full MSWC manifest50.

Luu y: hai Google Docs link chua doc duoc tu cong cu hien tai, nen ban nay chua the bam sat doan ban da viet tren Google Docs. Neu can sua truc tiep theo document do, hay bat quyen "Anyone with the link can view" hoac export/paste doan hien co.

## De xuat cau truc thesis

Nen giu cau truc gan voi PDF mau:

1. Declaration
2. Acknowledgements
3. Abstract
4. Chapter 1: Introduction
   - 1.1 Context and Motivation
   - 1.2 Problem Statement
   - 1.3 Objectives
   - 1.4 Research Questions
   - 1.5 Contributions
   - 1.6 Report Structure
5. Chapter 2: Background
   - Keyword Spotting
   - Few-shot Learning and Prototypical Networks
   - Open-set Recognition
   - Audio Features: MFCC, Mel Spectrogram, PCEN
   - Model Architectures: DSCNN-L and EdgeSpotFull T4
   - Metric Learning Losses: Triplet, SCAF, GE2E
   - Evaluation Metrics: FAR, FRR, ACC@FAR, AUC, EER, DET
6. Chapter 3: Methodology
   - System pipeline
   - Dataset pipeline
   - Training protocol
   - Evaluation protocol
   - Demo system
7. Chapter 4: Experiments and Results
   - Microset main evidence
   - Full MSWC 16-pipeline ablation
   - Shortlist experiments
   - Top500 recheck/preliminary evidence
   - Discussion
8. Chapter 5: Conclusion and Future Work

## Abstract - ban tieng Viet de dich

Do an nay nghien cuu bai toan few-shot open-set keyword spotting, trong do he thong can nhan dien cac tu khoa moi chi tu mot so luong nho mau enrollment, dong thoi phai tu choi cac am thanh khong thuoc tap tu khoa da dang ky. Khac voi keyword spotting closed-set truyen thong, bai toan nay khong chi yeu cau phan loai dung tu khoa, ma con yeu cau kiem soat false accept rate de giam truong hop unknown bi nhan nham thanh keyword.

He thong duoc xay dung theo huong embedding-based keyword spotting. Audio dau vao duoc chuan hoa, trich xuat dac trung MFCC hoac mel-PCEN, sau do dua qua encoder de tao embedding. Trong giai doan inference, moi keyword duoc bieu dien bang prototype tinh tu trung binh embedding cua cac mau support. Query audio duoc so khop voi cac prototype bang khoang cach L2 va duoc chap nhan hoac tu choi dua tren nguong open-set.

Do an danh gia nhieu thanh phan trong pipeline, bao gom hai backbone DSCNN-L va EdgeSpotFull T4, hai frontend MFCC va PCEN, va bon cach huan luyen Triplet, SCAF, GE2E, SCAF+GE2E. Ket qua Microset cho thay EdgeSpotFull T4 ket hop PCEN va SCAF+GE2E dat ACC@5%FAR = 86.12%, AUC = 95.61%, EER = 11.54%, va F1 = 82.41% tren GSC test100. O phan mo rong Full MSWC, thi nghiem 16 pipeline cho thay PCEN va GE2E co anh huong tich cuc ro nhat, dac biet voi DSCNN-L + PCEN + GE2E. Trong shortlist Full MSWC manifest20, DSCNN-L + PCEN + GE2E dat ACC@1%FAR = 82.10%, trong khi EdgeSpotFull T4 + PCEN + GE2E dat ACC@1%FAR = 79.58% voi so tham so nho hon dang ke.

Dong gop chinh cua do an la xay dung va danh gia mot pipeline few-shot open-set KWS co kha nang enroll tu khoa moi, so sanh co he thong cac ket hop model-feature-loss, va phat trien demo web ho tro enrollment, single detection, long-audio analysis, open-set testing va calibration. Ket qua cho thay viec can bang giua do chinh xac va do nho cua model la yeu to quan trong: DSCNN-L phu hop khi uu tien accuracy, trong khi EdgeSpotFull T4 phu hop hon voi huong edge/device nho gon.

## Chapter 1 - Introduction

### 1.1 Context and Motivation

Keyword Spotting (KWS) la bai toan phat hien mot hoac nhieu tu khoa muc tieu trong tin hieu am thanh. Day la thanh phan quan trong trong cac he thong giao tiep bang giong noi, chang han nhu tro ly ao, smart home, he thong dieu khien ranh tay, va wake-word detection. Trong cac ung dung nay, he thong thuong phai hoat dong voi do tre thap, tai nguyen gioi han, va phai tranh viec kich hoat sai khi nguoi dung khong noi tu khoa.

Nhieu he thong KWS truyen thong duoc thiet ke theo bai toan closed-set classification. Model duoc huan luyen tren mot tap keyword co dinh va khi nhan audio dau vao, no se chon mot trong cac lop da biet. Cach tiep can nay phu hop khi tap tu khoa khong thay doi, nhung chua du linh hoat cho truong hop nguoi dung muon them tu khoa moi bang vai mau giong noi. Neu moi lan them tu khoa moi deu phai thu thap du lieu lon va retrain model, he thong se kho trien khai trong thuc te.

Few-shot keyword spotting giai quyet van de nay bang cach hoc mot khong gian embedding, trong do cac mau cung tu khoa nam gan nhau va cac tu khoa khac nhau nam xa nhau. Khi nguoi dung enroll mot tu khoa moi, he thong chi can tinh prototype tu mot so mau support. Tuy nhien, trong moi truong thuc te, query audio khong phai luc nao cung thuoc cac tu khoa da enroll. Vi vay, he thong can co kha nang open-set rejection, tuc la tra ve unknown khi audio khong du gan voi bat ky prototype nao.

Dong luc cua do an la xay dung mot pipeline KWS co the vua ho tro keyword ca nhan hoa voi it mau, vua giam false accept trong open-set setting. Ngoai ra, do an cung quan tam den kha nang trien khai tren thiet bi tai nguyen han che, nen can so sanh giua model co accuracy cao va model nho gon phu hop voi edge/device.

### 1.2 Problem Statement

Bai toan duoc dinh nghia nhu sau. Cho mot tap keyword da enroll, moi keyword co mot so mau support. Voi mot audio query moi, he thong can dua ra mot trong hai quyet dinh:

- chap nhan query la mot keyword da enroll neu embedding cua query du gan voi prototype cua keyword do;
- tu choi query la unknown neu no khong du gan voi bat ky prototype nao hoac neu khoang cach giua top-1 va top-2 khong du ro rang.

Thach thuc chinh nam o viec so mau enrollment rat nho, cac tu ngan de nham lan voi nhau, va unknown audio co the rat gan voi keyword that. Neu nguong chap nhan qua de, he thong se co false accept cao. Neu nguong qua chat, he thong se reject ca keyword dung, lam tang false reject rate. Do do, muc tieu khong chi la tang keyword accuracy, ma con la toi uu trade-off giua false accept va false reject tai cac muc FAR co dinh.

### 1.3 Objectives

Muc tieu cua do an gom:

1. Xay dung pipeline few-shot open-set KWS dua tren embedding va prototype inference.
2. So sanh cac ket hop giua model architecture, audio frontend va loss function, bao gom DSCNN-L, EdgeSpotFull T4, MFCC, PCEN, Triplet, SCAF va GE2E.
3. Danh gia he thong bang protocol GSC few-shot open-set voi cac metric ACC@1%FAR, ACC@5%FAR, AUC, EER, FRR, F1 va DET curve.
4. Xac dinh pipeline phu hop cho hai muc tieu khac nhau: accuracy-oriented model va compact edge-oriented model.
5. Phat trien demo web de minh hoa quy trinh enrollment, detection, long-audio analysis, open-set testing va threshold calibration.

### 1.4 Research Questions

Do an tap trung tra loi cac cau hoi nghien cuu sau:

1. PCEN co giup cai thien few-shot open-set KWS so voi MFCC trong cung model va loss hay khong?
2. GE2E co phu hop hon Triplet loss khi inference su dung prototype/centroid hay khong?
3. SCAF+GE2E co luon tot hon GE2E don le hay khong, hay phu thuoc vao dataset va trong so loss?
4. DSCNN-L va EdgeSpotFull T4 the hien trade-off nhu the nao giua accuracy va so tham so?
5. Ket qua tren Microset co chuyen tiep nhu the nao khi mo rong sang Top500 va Full MSWC?

### 1.5 Contributions

Nhung dong gop chinh cua do an la:

1. Xay dung mot he thong few-shot open-set KWS end-to-end, tu xu ly audio, feature extraction, encoder embedding, prototype inference den demo web.
2. Thuc hien so sanh co he thong 16 pipeline tren Full MSWC phase-1, gom 2 architecture, 2 frontend va 4 loss settings.
3. Cho thay PCEN va GE2E la hai thanh phan co anh huong tich cuc nhat trong cac thi nghiem Full MSWC phase-1, dac biet voi DSCNN-L + PCEN + GE2E.
4. Xac nhan EdgeSpotFull T4 la huong model nho gon co gia tri cho edge/device, du DSCNN-L dat accuracy cao hon trong shortlist Full MSWC.
5. Cung cap demo co kha nang giai thich ket qua bang distance, threshold, margin, open-set rejection va long-audio error analysis.

### 1.6 Report Structure

Phan con lai cua bao cao duoc to chuc nhu sau. Chapter 2 trinh bay kien thuc nen ve keyword spotting, few-shot learning, open-set recognition, cac dac trung audio va metric danh gia. Chapter 3 mo ta pipeline de xuat, bao gom xu ly du lieu, model architecture, loss function, training protocol va inference protocol. Chapter 4 trinh bay ket qua thuc nghiem tren Microset, Top500 va Full MSWC, dong thoi phan tich DET curve va cac trade-off giua accuracy, FAR va model size. Chapter 5 tom tat ket luan, gioi han hien tai va huong phat trien tiep theo.

## Noi dung nen dua vao Background

### Keyword Spotting

Nen giai thich KWS la bai toan phat hien keyword trong audio ngan hoac stream. Can phan biet:

- closed-set KWS: dau vao luon bi gan vao mot lop da biet;
- open-set KWS: he thong co quyen tra ve unknown;
- few-shot KWS: keyword moi duoc them bang it mau support, khong retrain toan bo model.

### Prototypical / embedding-based inference

Nen viet theo y:

- Encoder bien audio thanh embedding.
- Moi keyword co prototype bang trung binh embedding cua support samples.
- Query duoc so voi cac prototype bang distance.
- Neu distance nho hon threshold thi accept, nguoc lai reject.

Cong thuc co the trinh bay:

```text
c_k = mean(f_theta(x_i)), x_i in support set of keyword k
y = argmin_k d(f_theta(x_query), c_k)
accept if min distance <= threshold
```

### Feature extraction: MFCC va PCEN

Nen trinh bay MFCC la baseline truyen thong, gon va pho bien trong speech tasks. PCEN nen duoc giai thich nhu mot frontend co kha nang chuan hoa nang luong theo kenh, giup on dinh hon voi khac biet am luong va noise. Trong ket qua Full MSWC, PCEN cai thien ro khi ket hop voi GE2E:

- DSCNN-L + GE2E: MFCC 72.30% -> PCEN 76.67% ACC@1%FAR.
- EdgeSpotFull T4 + GE2E: MFCC 69.31% -> PCEN 72.94% ACC@1%FAR.

### Loss functions

Triplet loss hoc bang quan he anchor-positive-negative, nhung khong truc tiep khop voi co che inference dung prototype trung binh. GE2E phu hop hon vi train theo centroid/prototype, gan hon voi inference few-shot. SCAF giup tang separation bang angular margin va sub-center, nhung khi ket hop voi GE2E khong phai luc nao cung tot hon GE2E don le; ket qua Full MSWC phase-1 cho thay SCAF+GE2E bi kem GE2E trong nhieu setting, co the do trong so loss chua toi uu hoac training schedule ngan.

## Noi dung Results/Discussion nen viet

### Microset

Microset nen duoc viet la evidence chinh cho thesis baseline vs proposed direction.

Ket qua quan trong:

- DSCNN-L + MFCC + Triplet test100: ACC@5%FAR = 80.54%, AUC = 91.22%, EER = 18.22%, F1 = 73.30%.
- EdgeSpotFull T4 + PCEN + SCAF test100: ACC@5%FAR = 85.21%, AUC = 95.69%, EER = 11.89%, F1 = 81.92%.
- EdgeSpotFull T4 + PCEN + SCAF+GE2E test100: ACC@5%FAR = 86.12%, AUC = 95.61%, EER = 11.54%, F1 = 82.41%.

Cach dien giai:

- EdgeSpotFull T4 + PCEN + SCAF+GE2E tang 5.58 diem phan tram ACC@5%FAR so voi DSCNN-L + MFCC + Triplet.
- F1 tang 9.11 diem phan tram.
- EER giam 6.68 diem phan tram.
- Ket qua nay ung ho viec chuyen tu baseline Triplet/MFCC sang huong embedding compact voi PCEN va metric learning loss phu hop hon.

### Full MSWC 16-pipeline phase-1

Phan nay nen viet la ablation de hieu anh huong tung thanh phan, khong nen xem la final training vi chi train 5 epoch, 150 episodes/epoch va manifest20.

Ket qua chinh:

- DSCNN-L + PCEN + GE2E la pipeline phase-1 tot nhat: ACC@1%FAR = 76.67%, ACC@5%FAR = 79.98%.
- EdgeSpotFull T4 + PCEN + GE2E la EdgeSpot pipeline tot nhat: ACC@1%FAR = 72.94%, ACC@5%FAR = 73.35%.
- SCAF+GE2E khong tot hon GE2E trong Full MSWC phase-1.

Cach dien giai:

- GE2E co loi the vi no huan luyen embedding theo centroid, gan voi cach inference prototype.
- PCEN co tac dung ro nhat khi ket hop voi GE2E.
- SCAF+GE2E can tuning trong so loss, learning rate hoac schedule dai hon truoc khi ket luan no kem ve ban chat.

### Full MSWC shortlist

Phan shortlist dung de chon huong train/evaluate tiep.

Manifest20 test100:

- DSCNN-L + PCEN + GE2E: ACC@1%FAR = 82.10 +/- 0.87, ACC@5%FAR = 86.05 +/- 0.66, AUC = 91.57 +/- 0.58, EER = 16.25 +/- 0.86.
- EdgeSpotFull T4 + PCEN + GE2E: ACC@1%FAR = 79.58 +/- 0.91, ACC@5%FAR = 83.06 +/- 0.82, AUC = 87.22 +/- 0.75, EER = 20.40 +/- 1.01.

Dien giai:

- DSCNN-L la accuracy-oriented candidate.
- EdgeSpotFull T4 la compact edge-oriented candidate, vi tham so it hon nhieu.
- Khong nen noi EdgeSpot tot hon DSCNN tren Full MSWC shortlist; nen noi no nho hon va van co gia tri trien khai.

### Top500

Top500 nen viet can than:

- Top500 epoch13 checkpoint local da re-evaluate duoc.
- test100: ACC@1%FAR = 85.62%, ACC@5%FAR = 88.79%, AUC = 95.34%, EER = 11.51%, F1 = 82.45%.
- Day la artifact co san va reproduce duoc.
- Epoch25 neu khong co checkpoint thi chi nen noi la historical/logged run, khong dung lam final claim.

## Goi y cau tieng Anh va ngu phap

### Tu/cum nen dung

| Y tieng Viet | Nen viet tieng Anh | Khong nen viet |
|---|---|---|
| mo rong tu Top500 sang Full MSWC | extend the experiment from Top500 to Full MSWC English | extend from top500full to Full MSWS |
| danh gia kha nang tong quat hoa | evaluate generalization ability | evaluate possible overally |
| gioi han 20 clips/word | cap the manifest at 20 clips per word | limit 20clips/word |
| so sanh 16 pipeline | compare 16 pipeline configurations | compare 16 pipeline |
| khi ket hop voi | when combined with | when come up with |
| dat ket qua | achieves / obtains | archieve |
| it tham so hon | has significantly fewer parameters | significantly lighter compare with |
| ket qua so bo | preliminary result | first result good |
| khong nen overclaim | should be interpreted with caution | prove everything |

### Mau cau cho Abstract

```text
This thesis investigates few-shot open-set keyword spotting, where a system must recognize user-enrolled keywords from only a few examples while rejecting non-enrolled speech as unknown.
```

```text
Unlike conventional closed-set KWS, the target system is required not only to classify known keywords but also to control false accepts under fixed FAR operating points.
```

```text
The proposed pipeline uses an embedding encoder and prototype-based inference, where each keyword is represented by the mean embedding of its support examples.
```

```text
Experiments compare DSCNN-L and EdgeSpotFull T4 backbones, MFCC and PCEN frontends, and Triplet, SCAF, GE2E, and SCAF+GE2E losses.
```

```text
The results show that PCEN and GE2E are the most consistent contributors in the Full MSWC ablation, while EdgeSpotFull T4 remains valuable as a compact edge-oriented model.
```

### Mau cau cho Objectives

```text
The first objective is to build an end-to-end few-shot open-set KWS pipeline based on embedding learning and prototype inference.
```

```text
The second objective is to evaluate how model architecture, audio frontend, and metric-learning loss affect open-set performance.
```

```text
The final objective is to develop a demo system that exposes enrollment, detection, calibration, and long-audio analysis in an interpretable way.
```

### Mau cau cho Discussion

```text
This improvement suggests that GE2E is better aligned with prototype-based inference than Triplet loss, because both training and inference rely on class centroids.
```

```text
PCEN provides a stronger frontend for this setting because it normalizes channel energy and helps reduce sensitivity to loudness variation.
```

```text
Although DSCNN-L achieves higher accuracy in the shortlist experiments, EdgeSpotFull T4 uses substantially fewer parameters and is therefore more suitable for edge-oriented deployment.
```

```text
The SCAF+GE2E setting should not be considered universally worse than GE2E; rather, the current results indicate that the hybrid loss requires further tuning under the Full MSWC schedule.
```

### Loi ngu phap can sua trong bai

1. Dung `experiments`, khong dung `experient`.
2. Dung `split`, khong dung `spit`.
3. Dung `clips`, khong dung `clip` khi noi so nhieu: `6.62M clips`.
4. Dung `pipelines` khi so nhieu: `16 pipeline configurations`.
5. Dung `achieves`, khong dung `archieve`.
6. Dung `compared with`, khong dung `compare with` sau tinh tu: `130k parameters compared with 412k`.
7. Dung `when combined with`, khong dung `when combined GE2E`.
8. Viet `ACC@1%FAR`, `ACC@5%FAR` nhat quan.
9. Viet `Full MSWC English`, khong viet `Full MSWS`.
10. Viet `This result should be interpreted as preliminary evidence`, neu artifact/training chua final.

## Doan Acknowledgements sua gon bang tieng Anh

Neu can sua phan Acknowledgements, co the dung ban sau:

```text
First and foremost, I would like to express my sincere gratitude to my supervisor, Dr. Tran Hoang Tung, for his guidance, insightful feedback, and continuous support throughout my internship.

I would also like to thank Dr. Tran Giang Son for providing access to the ICTLab server, which made the computational experiments in this project possible.

I am deeply grateful to my family for their encouragement and emotional support, which helped me stay motivated during the project.

Finally, I would like to thank my friends for their support in reviewing and improving this thesis.
```

## Claim hygiene cho thesis

Nen viet:

- `Microset is used as the main controlled evidence for architecture selection.`
- `Full MSWC phase-1 is used as an ablation study to compare pipeline combinations.`
- `Top500 epoch13 is a reproducible local checkpoint result.`
- `Top500 epoch25 should be described only as a logged historical run unless the checkpoint artifact is available.`

Khong nen viet:

- `This project fully reproduces EdgeSpot.`
- `SCAF+GE2E is always the best loss.`
- `Full MSWC proves EdgeSpot is better than DSCNN.`
- `Open-set UI sampled evaluation replaces GSC test100.`
