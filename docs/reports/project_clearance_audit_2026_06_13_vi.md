# Bao cao clear du an KWS - UI policy, thesis, va viec can lam

Ngay audit: 2026-06-13  
Workspace: `D:\Downloads\DoAnTotNghiep`

## 1. Ket luan ngan

Du an khong hong, nhung dang bi tron ba tang bang chung:

1. **Benchmark nghien cuu chinh**: `gsc_edgespot_exact dev30/test100`, co target FAR, lap 30/100 runs. Day la bang chung dung de viet thesis va so sanh paper.
2. **Demo UI sampled open-set**: enroll vai tu, lay mau GSC nho, thu threshold/guard/per-class. Day la cong cu debug/demo, khong thay the benchmark.
3. **Long-audio/streaming demo**: minh hoa workflow thuc te, nhung chua phai benchmark streaming chinh thuc vi chua co false alarm per hour, latency, miss rate tren audio lien tuc.

Cam giac "nguong tung lop khong hieu qua" va "chan cac tu gan nhau khong hieu qua" la co co so. Hai chuc nang nay co the co ich de debug, nhung khong nen dat lam mac dinh chinh cho demo/thesis neu chua calibrate bang negative/unknown that.

## 2. Bang chung da kiem tra

### 2.1. Code backend

File: `src/demo/api_server.py`

- `DetectionPolicy` gom `threshold`, `use_per_class`, `close_word_guard`, `accept_margin`.
- `build_detection_policy(...)`:
  - neu `close_word_guard=true` va UI gui `accept_margin`, backend dung margin do;
  - neu guard bat nhung khong gui margin, backend dung mac dinh `0.05`;
  - neu guard tat, backend dung `accept_margin=0.0`.
- `score_embedding(...)`:
  - tinh khoang cach query toi prototype/cac keyword profile;
  - neu `use_per_class=true`, threshold cua best label lay tu enrollment profile hoac `proto_thresholds`;
  - accept khi `best_dist <= effective_threshold` va `margin >= accept_margin`;
  - margin la `second_dist - best_dist`.
- `/api/open-set/test` danh gia sampled GSC voi known/unknown words.
- `/api/open-set/calibrate` grid-search `threshold`, `accept_margin`, `use_per_class` va chon `best_balanced`, `best_open_set`, `best_keyword`.

File: `src/streaming/enrollment.py`

- Moi keyword co `KeywordProfile.threshold`.
- Threshold duoc tinh bang:
  - `mean(support_dists) + alpha * std(support_dists)`;
  - clamp vao `[0.35, 1.25]`;
  - neu co impostor waveforms thi lay them gioi han theo impostor quantile.
- Trong flow demo hien tai, threshold nay phu thuoc rat manh vao so luong/muc da dang cua mau enrollment.

### 2.2. Code UI

File: `src/demo/ui/src/App.tsx`

- Open-set default hien tai:
  - `openThreshold = 0.3`
  - `openPerClass = false`
  - `openGuard = true`
  - `openMargin = 0.05`
- UI co nut calibration va `applyCalibration(row)` de copy threshold/per-class/guard/margin tu row tot nhat vao form.
- Long-audio default van bat `longPerClass=true`, `longGuard=true`.
- Single detection co toggle per-class va guard.

Verification:

- `npm.cmd run typecheck`: pass.
- `npm.cmd run build`: pass.
- `npm run typecheck` truc tiep bi PowerShell execution policy chan `npm.ps1`, khong phai loi TypeScript.

## 3. Danh gia chuc nang "nguong tung lop"

### 3.1. No dang lam gi

Nguong tung lop dat moi tu khoa mot threshold rieng. Tu nao co support embeddings phan tan hon se co threshold rong hon; tu nao support embeddings chat hon se co threshold chat hon.

### 3.2. Vi sao nghe hop ly

Trong few-shot KWS, moi tu co do kho khac nhau. Cac tu ngan hoac phu am yeu co the co embedding bien dong hon. Threshold rieng co the giup he thong khong ep tat ca keyword dung mot nguong global.

### 3.3. Vi sao hien tai co the khong hieu qua

1. **It mau enrollment**: neu moi tu chi co 3-10 mau, `mean + 2*std` la uoc luong rat nhieu nhieu. Chi can mot mau enroll lech la threshold bi keo rong.
2. **Khong chac co negative/impostor**: threshold tot cho open-set phai nhin ca khoang cach cua unknown/negative. Neu chi nhin support cua chinh keyword thi no chi biet "noi bo tu nay phan tan ra sao", khong biet unknown co de chen vao hay khong.
3. **Clamp lam mat khac biet**: `[0.35, 1.25]` giup an toan, nhung neu nhieu tu cung bi day ve floor/ceil thi per-class khong con nhieu y nghia.
4. **Threshold UI khong giong threshold benchmark**: `gsc_edgespot_exact` tim threshold theo target FAR tren episode/eval set. Demo per-class threshold la heuristic enrollment-time. Hai cai nay khong nen coi la tuong duong.

### 3.4. Ket luan policy

- Khong nen coi per-class threshold la chuc nang "tang accuracy" mac dinh.
- Nen doi vai tro thanh **che do thu nghiem/advanced**.
- Mac dinh demo nen de `per-class OFF`, sau do dung `/api/open-set/calibrate` de xem row nao tot hon.
- Chi nen bat per-class neu calibration tren known/unknown that cho thay balanced score tot hon global threshold.

## 4. Danh gia chuc nang "chan cac tu gan nhau"

### 4.1. No dang lam gi

Close-word guard khong thuc su biet hai tu co gan am hoc hay khong. No chi kiem tra margin giua ung vien gan nhat va gan nhi:

`margin = distance(top2) - distance(top1)`

Neu margin nho hon `accept_margin`, backend reject ve unknown.

### 4.2. Vi sao nghe hop ly

Neu query nam giua hai prototype gan nhau, top-1 khong dang tin. Reject khi top1/top2 qua gan co the giam nham lan giua cac keyword tuong tu.

### 4.3. Vi sao hien tai co the khong hieu qua

1. **Khong phai phonetic guard**: no khong dung phone/phoneme/edit distance. "Tu gan nhau" o day chi la gan trong embedding, va embedding co the gan vi noise/speaker, khong phai vi phat am.
2. **Phu thuoc tap enrolled**: neu chi enroll it tu, top2 co the rat xa, margin luon pass. Neu enroll nhieu tu gan nhau, margin co the qua chat va reject ca true positive.
3. **Unknown van co the pass**: mot unknown co the rat gan mot prototype va xa prototype thu hai, luc do margin lon nen guard khong chan.
4. **Accept margin 0.05 la heuristic**: no can calibration rieng cho tung model/enrollment set. Mot so co dinh khong on cho moi checkpoint.

### 4.4. Ket luan policy

- Close-word guard nen la **margin rejection**, khong nen mo ta nhu mot bo "chan tu gan am" chinh xac.
- Mac dinh nen de `guard OFF` hoac `margin=0.0` trong workflow binh thuong.
- Neu user bam calibration va row tot nhat co `accept_margin > 0`, luc do moi bat guard.
- Trong thesis, day la engineering/debug policy, khong phai dong gop nghien cuu chinh.

## 5. De xuat clear UI/demo

### 5.1. Luong demo nen dung

1. Chon checkpoint.
2. Enroll GSC preset hoac upload mau.
3. Chay calibration open-set.
4. Bam apply row tot nhat.
5. Chay open-set test/single/long audio bang policy da calibrate.

### 5.2. Default nen sua

De giam roi:

- Open-set:
  - threshold: `0.3` tam duoc.
  - per-class: `OFF`.
  - guard: `OFF`.
  - accept margin: `0.0`.
  - Hien ro "Calibration recommended".
- Long-audio:
  - per-class: `OFF` mac dinh.
  - guard: `OFF` mac dinh.
  - neu co calibration row, cho nut "Use calibrated policy".
- Single detection:
  - giu controls nhung dua per-class/guard vao Advanced.

### 5.3. Ten hien thi nen doi

- `Nguong tung lop` -> `Nguong rieng theo tu (thu nghiem)`.
- `Chan tu gan nhau` -> `Reject khi top-1/top-2 qua sat`.
- `Accept margin` -> `Khoang cach toi thieu giua top-1 va top-2`.

### 5.4. Diem can giu

- Calibration table rat huu ich. Nen giu va lay no lam trung tam UI.
- False accept/known miss review rat huu ich de giai thich loi.
- Top-3 candidates, distance, threshold, margin nen giu vi giup debug.

## 6. Hien trang ket qua thuc nghiem

### 6.1. Fixed 16-pipeline cap620

Run: `colab_mswc_cap620_flac_16pipe_e40_ep150_20260611_154517`

Bang chung chinh ban dau:

- Best overall: `DSCNN-L + PCEN + GE2E`
  - ACC@1%FAR: `82.34 +/- 1.19`
  - AUC: `92.42 +/- 0.54`
  - EER: `14.89 +/- 0.84`
  - F1: `77.75 +/- 1.15`
- Best compact fixed: `EdgeSpotFull T4 + PCEN + GE2E`
  - ACC@1%FAR: `79.98 +/- 0.98`
  - AUC: `87.23 +/- 0.75`
  - EER: `20.23 +/- 0.96`
  - F1: `70.68 +/- 1.23`
- `EdgeSpotFull T4 + PCEN + Triplet` co AUC/EER/F1 tot trong compact fixed, nhung ACC@1%FAR kem GE2E nhe.
- SCAF/SCAF+GE2E bi collapse tren nhieu hang cap620.

### 6.2. Development run moi

Run: `colab_mswc_cap620_development_20260612_050614`

Bang chung moi hon:

- Best accuracy: `DSCNN-L + PCEN + GE2E, ep300 composite`
  - test100 FAR1 ACC: `86.36 +/- 1.29`
  - AUC: `95.21 +/- 0.45`
  - EER: `11.32 +/- 0.78`
  - F1: `82.73 +/- 1.11`
  - FAR5 ACC: `89.93 +/- 0.65`
- Best compact: `EdgeSpotFull T4 + PCEN + GE2E, ep300 composite`
  - test100 FAR1 ACC: `82.87 +/- 1.22`
  - AUC: `92.41 +/- 0.44`
  - EER: `14.82 +/- 0.70`
  - F1: `77.85 +/- 0.97`
  - FAR5 ACC: `86.76 +/- 0.59`
- `EdgeSpotFull T4 + PCEN + Triplet hard` collapse:
  - ACC: `69.10 +/- 0.15`
  - AUC: `53.40 +/- 0.48`
  - EER: `47.84 +/- 0.62`
  - F1: `39.99 +/- 0.60`

### 6.3. Claim voi EdgeSpot-4 paper

Moc dang dung trong thesis: EdgeSpot-4 paper bao cao `82.0% ACC@1%FAR`.

Claim nen viet:

- `DSCNN-L + PCEN + GE2E` da vuot moc paper ve mean (`86.36`), nhung day la model lon hon EdgeSpot-4, khong phai compact edge model.
- `EdgeSpotFull T4 + PCEN + GE2E` moi dat `82.87 +/- 1.22`, cao hon `82.0` ve mean nhung chen lech chi `+0.87` diem va nam trong do lech. Nen viet than trong: **competitive and slightly above the reported EdgeSpot-4 mean under our protocol**, khong viet "vuot dut khoat".
- Chua chay KD trong development run (`RUN_KD=0`), nen khong claim tai lap day du recipe paper EdgeSpot neu paper dung knowledge distillation.

## 7. Hien trang thesis/paper

### 7.1. File thesis dang co

1. `docs/thesis/Do_An_KWS_final_vi_2026_06_12.md`
   - Ban final draft ngay 2026-06-12.
   - Dang dua tren fixed 16-pipeline.
   - Chua cap nhat development run moi `86.36` va `82.87`.
   - Phan EdgeSpot-4 con noi compact EdgeSpotFull T4 chua vuot paper theo fixed run; can cap nhat lai voi claim than trong.

2. `docs/thesis/Do_An_KWS_final_vi_2026_06_12.docx`
   - Ban Word tu draft tren.
   - Neu cap nhat thesis MD thi can regenerate docx.

3. `docs/thesis/Do_An_KWS_completed_vi_2026_06_04.md`
   - Ban cu hon, muc luc kha ro, co cac chuong Dataset, Model, Pipeline, Results, Demo.
   - Co the dung de lay khung, nhung so lieu da cu.

4. `docs/thesis/cap620_16_pipeline_scientific_chapter_vi_2026_06_12.md`
   - Chuong thuc nghiem cap620 16 pipeline rat huu ich.
   - Chua cap nhat development run moi.

### 7.2. Muc luc hien co cua ban final 2026-06-12

1. Loi cam on
2. Tom tat
3. Danh muc thuat ngu viet tat
4. Chuong 1. Gioi thieu
5. Chuong 2. Co so ly thuyet va cong trinh lien quan
6. Chuong 3. Thiet ke he thong va phuong phap
7. Chuong 4. Thuc nghiem
8. Chuong 5. Ket qua va thao luan
9. Chuong 6. So sanh voi EdgeSpot-4 paper
10. Chuong 7. Demo system va trien khai
11. Chuong 8. Ket luan va huong phat trien
12. Threats to Validity
13. Phu luc A. Reproducibility Checklist
14. Tai lieu tham khao

Khung nay dung, nhung can sap lai de "paper/thesis chuan" hon: tach dataset, method, experimental setup, results, discussion ro hon.

## 8. Muc luc thesis de xuat ban nop

### Phan dau

1. Bia va thong tin sinh vien/giang vien huong dan
2. Loi cam on
3. Tom tat tieng Viet
4. Abstract tieng Anh
5. Danh muc tu viet tat
6. Danh muc bang
7. Danh muc hinh

### Chuong 1. Gioi thieu

1.1. Boi canh Keyword Spotting  
1.2. Bai toan few-shot open-set KWS  
1.3. Thach thuc: enrollment it mau, false accept, domain shift MSWC -> GSC  
1.4. Muc tieu nghien cuu va cau hoi nghien cuu  
1.5. Dong gop cua do an  
1.6. Cau truc bao cao

### Chuong 2. Nen tang va cong trinh lien quan

2.1. Keyword spotting closed-set va open-set  
2.2. Few-shot learning va prototype inference  
2.3. Frontend MFCC, Mel, PCEN  
2.4. Backbone DSCNN va EdgeSpot/EdgeSpotFull T4  
2.5. Metric learning losses: Triplet, GE2E, SCAF, SCAF+GE2E  
2.6. Knowledge distillation cho compact KWS  
2.7. Cac metric FAR, FRR, EER, AUC, ACC@FAR, F1  
2.8. Tong ket khoang trong nghien cuu

### Chuong 3. Phuong phap de xuat

3.1. Tong quan pipeline embedding + prototype + threshold  
3.2. Xu ly audio va feature extraction  
3.3. Kien truc DSCNN-L  
3.4. Kien truc EdgeSpotFull T4  
3.5. Episodic training va cau hinh support/query  
3.6. Objective: Triplet, GE2E, SCAF, SCAF+GE2E  
3.7. Prototype inference va open-set thresholding  
3.8. Phan biet threshold benchmark va threshold demo UI  
3.9. Checkpoint selection theo dev metric tong hop  
3.10. Demo system: enrollment, detection, long audio, calibration

### Chuong 4. Thiet lap thuc nghiem

4.1. Dataset MSWC English cap620 FLAC  
4.2. Dataset GSC evaluation va open-set split  
4.3. Cau hinh fixed 16-pipeline  
4.4. Cau hinh development run ep300 composite  
4.5. Cau hinh evaluation `gsc_edgespot_exact`  
4.6. Test100 FAR1/FAR5 va confidence interval  
4.7. Moi truong huan luyen Colab/server  
4.8. Reproducibility: script, run id, artifact

### Chuong 5. Ket qua thuc nghiem

5.1. Ket qua fixed 16-pipeline day du  
5.2. Phan tich frontend: PCEN so voi MFCC  
5.3. Phan tich loss: GE2E, Triplet, SCAF collapse  
5.4. Phan tich backbone: DSCNN-L so voi EdgeSpotFull T4  
5.5. Development run: ep300 composite va ket qua moi  
5.6. Best accuracy: DSCNN-L + PCEN + GE2E  
5.7. Best compact: EdgeSpotFull T4 + PCEN + GE2E  
5.8. Failure cases: SCAF collapse, hard-triplet collapse  
5.9. So sanh voi Top500/Microset nhu evidence bo sung, khong tron protocol

### Chuong 6. So sanh voi EdgeSpot-4 paper

6.1. Metric va moc paper EdgeSpot-4  
6.2. Khac biet protocol va han che reproduction  
6.3. So sanh best overall project voi EdgeSpot-4  
6.4. So sanh compact EdgeSpotFull T4 voi EdgeSpot-4  
6.5. Vai tro cua KD va vi sao chua claim reproduce paper  
6.6. Claim hop le va claim khong nen viet

### Chuong 7. Demo va trien khai

7.1. Kien truc backend/frontend demo  
7.2. Model switcher va artifact loading  
7.3. Enrollment va prototype building  
7.4. Single detection  
7.5. Long-audio analysis  
7.6. Open-set sampled calibration  
7.7. Danh gia policy per-class threshold va margin guard  
7.8. Gioi han cua demo-level evaluation

### Chuong 8. Thao luan

8.1. Vi sao PCEN + GE2E tot  
8.2. Vi sao compact model kho vuot paper neu thieu KD  
8.3. Vi sao SCAF can ablation rieng  
8.4. Threshold calibration trong open-set KWS  
8.5. Threats to validity  
8.6. Bai hoc engineering va reproducibility

### Chuong 9. Ket luan va huong phat trien

9.1. Ket luan theo tung cau hoi nghien cuu  
9.2. Dong gop dat duoc  
9.3. Han che  
9.4. Huong phat trien: episode budget, KD, tuned SCAF, hard episode mining, streaming benchmark

### Phu luc

A. Lenh tai lap thuc nghiem  
B. Bang metric day du 16 pipeline  
C. Bang development run  
D. Artifact/checkpoint can tai ve  
E. Huong dan chay demo  
F. Mo ta tham so UI va calibration

## 9. Nhung phan thesis dang thieu/can bo sung

### P0 - Bat buoc truoc khi nop

1. Cap nhat so lieu moi vao tom tat, ket qua, ket luan:
   - DSCNN-L + PCEN + GE2E: `86.36 +/- 1.29` ACC@1%FAR.
   - EdgeSpotFull T4 + PCEN + GE2E: `82.87 +/- 1.22` ACC@1%FAR.
2. Sua claim EdgeSpot-4:
   - khong con viet "compact chua vuot paper" nhu fixed run nua;
   - viet "compact moi cao hon mean paper nhung chua phai reproduction va chenh lech trong sai so".
3. Them muc phan biet:
   - benchmark test100 la evidence chinh;
   - UI open-set sampled chi la demo/debug.
4. Them bang artifact:
   - checkpoint best accuracy;
   - checkpoint best compact;
   - run id;
   - file ket qua;
   - lenh run/eval.
5. Them failure analysis:
   - SCAF collapse tren cap620;
   - Triplet hard collapse trong development run.

### P1 - Nen lam de thesis manh hon

1. Them Abstract tieng Anh.
2. Them mot bang "research questions -> evidence -> conclusion".
3. Them bang "claim hop le/khong hop le" voi EdgeSpot-4.
4. Them muc calibration trong demo: per-class/margin la engineering policy, khong phai metric final.
5. Them threat-to-validity ve:
   - khac protocol paper;
   - checkpoint selection noise;
   - Colab/session;
   - dataset cap vs episode budget;
   - mixed evidence Microset/Top500/cap620.

### P2 - Neu con thoi gian

1. Chay mot ablation UI nho cho per-class/guard:
   - per-class ON/OFF;
   - guard margin 0/0.02/0.05/0.08/0.10;
   - 3 seeds;
   - bao cao keyword acc, unknown reject acc, balanced score.
2. Them figure DET/ROC cho best overall va best compact.
3. Them parameter/MAC table neu co script tinh MAC.
4. Regenerate Word/PDF tu MD.

## 10. Thu tu dau viec de thoat roi

1. **Khoa bang chung chinh**: dung development run moi lam result chinh, fixed 16-pipeline lam ablation nen.
2. **Don claim**: moi claim phai gan voi run id va protocol.
3. **Don UI**: default global threshold, per-class OFF, guard OFF; calibration la workflow chinh.
4. **Viet lai thesis theo muc luc de xuat**: khong viet lan man theo lich su thu nghiem.
5. **Xuat Word sau cung**: chi xuat sau khi MD da cap nhat so lieu va claim.

## 11. Khuyen nghi quyet dinh ngay

- Per-class threshold: **giu nhung dua vao Advanced/Experimental, default OFF**.
- Close-word guard: **doi ten thanh margin rejection, default OFF neu chua calibration**.
- Open-set calibration: **dua thanh workflow chinh cua UI**.
- Thesis: **cap nhat ban final 2026-06-12 bang development run 2026-06-12, khong viet tiep tren ban cu ma khong sua claim**.
- Paper claim: **co the noi compact EdgeSpotFull T4 moi da canh tranh va nhinh hon mean EdgeSpot-4, nhung khong claim reproduce/vuot dut khoat**.

