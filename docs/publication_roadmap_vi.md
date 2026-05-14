# Roadmap Dua De Tai KWS Len Muc Paper

## Muc Tieu

Muc tieu thuc te la dua project tu do an/demo thanh mot pipeline nghien cuu co
the nop workshop/main track lien quan speech:

- muc gan: INTERSPEECH / ICASSP workshop hoac main track neu ket qua du manh;
- muc mo rong: AAAI / IJCAI / ACL / EMNLP neu dong gop du tong quat va co
  novelty ro.

Khong claim A* neu chi co demo hoac chi train lai mot model. Paper phai co:
benchmark chuan, baseline manh, ablation, ket qua lap lai duoc, va dong gop moi.

## Huong Dong Gop Nen Claim

Dong gop nen tap trung vao **few-shot open-set KWS trong dieu kien enrollment it
mau va gioi han tai nguyen**:

1. Reproduce/gan dung EdgeSpot-style encoder tren giao thuc GSC exact.
2. Chuan hoa data profile: Microset official, Top500 full, va neu co may truong
   thi full MSWC English.
3. Them calibration cho open-set: impostor bank, per-keyword threshold,
   support uncertainty, multi-prototype.
4. Do streaming sau khi static/open-set on dinh: false alarm/hour, miss rate,
   latency, duplicate rate.
5. Ghi nhan ket qua DSCNN la baseline/negative result, khong coi day la that bai.

## Data Profiles Va Cach Claim

### `microset_en`

Dung khi Colab/Drive khong du dung luong. Day la MSWC Microset chinh thuc cua
MLCommons.

Duoc claim:

- official Microset experiment;
- pipeline validation trong dieu kien tai nguyen han che;
- ablation nhanh cho loss/model/calibration.

Khong duoc claim:

- reproduce EdgeSpot Top500/full MSWC;
- ket qua dai dien cho full MSWC.

### `top500_full`

Dung cho ket qua nghiem tuc hon. Profile nay la MSWC English Top500 train/val,
all clips per word, cache `mswc_en_wav_top500_full`.

Duoc claim:

- Top500 full-clips experiment;
- EdgeSpot-style reproduction attempt;
- baseline va proposed method tren benchmark chuan.

Can ghi ro:

- khong phai full MSWC English 38k words;
- neu khong include eval words thi chi extract 450 train + 50 val words.

## Ma Tran Thi Nghiem Toi Thieu

Chay theo thu tu:

1. `dscnn_triplet_baseline`
   Baseline cu de so sanh va ghi negative result.

2. `bcresnet_fs_scaf`
   Baseline kien truc nhe, khong attention.

3. `edgespot_full_t4_scaf`
   Reproduction EdgeSpot-style khong GE2E/KD.

4. `edgespot_full_t4_scaf_ge2e`
   Baseline chinh cua project hien tai.

5. `edgespot_full_t4_scaf_ge2e + calibration`
   Ung vien dong gop moi. Chi claim proposed method neu vuot baseline ro.

6. `kd_scaf_ge2e`
   Stretch phase. Chi claim neu teacher/projection duoc kiem chung, khong dung
   random projection head de claim khoa hoc.

## Benchmark Bat Buoc

Dung `gsc_edgespot_exact`:

- GSC-dev: chon checkpoint/calibration;
- GSC-test: final report, 100 runs;
- true `_silence_` tu `_background_noise_`, khong dung `marvin` lam silence;
- k-shot: 10 la moc EdgeSpot; co the bao them 5-shot neu lien quan de tai.

Metric chinh:

- `ACC@1% FAR`;
- `ACC@5% FAR`;
- `FRR@5% FAR`;
- `AUC`;
- `EER`;
- `keyword_acc`;
- per-word metrics/confusion matrix.

## Tieu Chi Dat

Top500 full:

- reproduction acceptable: `ACC@1% FAR >= 78%`;
- reproduction success: `ACC@1% FAR >= 80%`;
- stretch: gan `82%` neu pipeline du manh.

Proposed method:

- hon reproduction/baseline it nhat `+2 percentage points` tren `ACC@1% FAR`,
  hoac giam ro `FRR@5% FAR`;
- khong tang false alarm mot cach khong kiem soat.

## Cach Viet Paper

Title direction:

> Calibrated Few-Shot Open-Set Keyword Spotting under Resource-Constrained
> Enrollment

Abstract nen gom:

- bai toan: user chi co 3-10 mau moi tu;
- kho: open-set rejection va deployment mismatch;
- phuong phap: EdgeSpot-style encoder + GE2E/SCAF + calibration;
- benchmark: Microset/Top500 full + GSC exact;
- ket qua: so voi DSCNN/BCResNet/EdgeSpot-style baseline;
- gioi han: full MSWC/streaming la phase tiep theo neu chua xong.

## Khong Nen Lam

- Khong claim 90-95% neu benchmark strict chua chung minh.
- Khong so sanh voi EdgeSpot paper neu data profile khac ma khong ghi ro.
- Khong chon threshold tren GSC-test.
- Khong dung checkpoint `latest.pt` thay cho `best.pt`.
- Khong train tiep checkpoint degrade roi bao la model khong tot.

## Viec Can Lam Moi Lan Chay Colab

1. Pull code moi nhat.
2. Chon `DATA_PROFILE` trong notebook 03.
3. Chay data setup.
4. Chay `scripts/mswc_data_report.py`.
5. Smoke train 1 epoch/5 episodes.
6. Train main.
7. Evaluate GSC-dev neu can selection.
8. Evaluate GSC-test 100 runs.
9. Chay `scripts/research_readiness.py`.
10. Cap nhat bang ket qua bang `scripts/make_research_tables.py`.
