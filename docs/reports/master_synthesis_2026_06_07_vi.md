# Tổng hợp toàn bộ thực nghiệm KWS Few-Shot Open-Set (đến 2026-06-07)

Tài liệu này gom toàn bộ quá trình: thử các tổ hợp, chọn shortlist, nâng hạn mức dữ liệu, kiểm thử, suy luận bão hòa, và trạng thái server/Colab. Mọi số liệu lấy từ log/JSON thật.

## 0. Cách đọc số liệu trong log (quan trọng)

Mỗi lần eval in một khối kết quả cuối. Các con số FAR nằm như sau:

- `Open-set ACC` = ACC tại operating point của lượt eval đó.
  - Nếu khối có `FRR@5.0%FAR` thì `Open-set ACC` = ACC@5%FAR.
  - Nếu khối có `FRR@1.0%FAR` thì `Open-set ACC` = ACC@1%FAR.
- `AUC`, `EER`, `F1` không phụ thuộc ngưỡng nên giống nhau ở cả hai lượt.
- Trong các script eval chạy 2 lượt (`--target-far 0.05` rồi `0.01`), sẽ có 2 khối; đọc đúng khối theo dòng `FRR@...`.

## 1. Thiết kế thí nghiệm: các thành phần đã thử

Bài toán: few-shot open-set KWS. Train encoder trên MSWC English, đánh giá trên Google Speech Commands v2 theo giao thức `gsc_edgespot_exact` (k=10, test100 = 100 lần lặp, mỗi lần 11 từ khóa + 25 từ lạ + lớp `_silence_`).

Các trục đã thử:

- Backbone: `DSCNN-L` (412,900 tham số) và `EdgeSpotFull T4` (130,598 tham số, nhỏ hơn ~3.2 lần).
- Frontend: `MFCC` và `PCEN` (mel + Per-Channel Energy Normalization).
- Loss: `Triplet`, `SCAF` (Sub-center ArcFace), `GE2E`, `SCAF+GE2E`, và `KD` (knowledge distillation từ Wav2Vec2 teacher: `kd_scaf`).

## 2. Bước 1 — Microset (mốc kiến trúc đã khóa)

Dữ liệu nhỏ (31 từ khóa, MSWC Microset official). Mục đích: thử nhanh nhiều cấu hình để chọn hướng.

GSC test100:

| Mô hình | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 | KW-ACC |
|---|---:|---:|---:|---:|---:|---:|
| DSCNN-L + Triplet | - | 80.54 | 91.22 | 18.22 | 73.30 | 68.39 |
| EdgeSpotFull T4 + SCAF | 84.64 | 85.21 | 95.69 | 11.89 | 81.92 | 74.52 |
| EdgeSpotFull T4 + SCAF+GE2E (KHÓA) | 84.61 | 86.12 | 95.61 | 11.54 | 82.41 | 77.66 |

Suy luận: Triplet kém; SCAF và SCAF+GE2E rất tốt (AUC ~95.6). Chọn `EdgeSpotFull T4 + SCAF+GE2E` làm cấu hình kiến trúc chính thức của thesis. Đây là mốc đã khóa, không tinh chỉnh theo test100 nữa.

## 3. Bước 2 — Sàng lọc 16 tổ hợp trên Full MSWC (phase-1)

Ma trận đầy đủ = 2 backbone × 2 frontend × 4 loss = **16 tổ hợp**:
`{DSCNN-L, EdgeSpotFull T4} × {MFCC, PCEN} × {Triplet, SCAF, GE2E, SCAF+GE2E}`.

Dữ liệu: Full MSWC capped 20 file/từ, train ngắn (5 epoch × 150 episode) chỉ để xếp hạng xu hướng. Số đo: GSC-dev (phase-1, 3 run/checkpoint), không phải test100.

Bảng đầy đủ 16 tổ hợp (GSC-dev phase-1):

| # | Backbone | Frontend | Loss | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | DSCNN-L | MFCC | Triplet | 69.30 | 67.31 | 57.93 | 44.57 | 43.18 |
| 2 | DSCNN-L | MFCC | SCAF | 70.72 | 68.26 | 52.03 | 48.85 | 39.00 |
| 3 | DSCNN-L | MFCC | GE2E | 72.30 | 73.78 | 78.59 | 28.37 | 60.69 |
| 4 | DSCNN-L | MFCC | SCAF+GE2E | 69.24 | 66.83 | 52.04 | 48.71 | 39.16 |
| 5 | DSCNN-L | PCEN | Triplet | 72.24 | 72.67 | 73.71 | 33.38 | 54.98 |
| 6 | DSCNN-L | PCEN | SCAF | 70.02 | 67.67 | 52.32 | 49.15 | 38.74 |
| 7 | **DSCNN-L** | **PCEN** | **GE2E** | **76.67** | **79.98** | **85.89** | **22.60** | **67.68** |
| 8 | DSCNN-L | PCEN | SCAF+GE2E | 70.11 | 67.85 | 50.05 | 50.11 | 37.86 |
| 9 | EdgeSpot T4 | MFCC | Triplet | 69.00 | 66.35 | 48.86 | 50.78 | 37.20 |
| 10 | EdgeSpot T4 | MFCC | SCAF | 69.50 | 67.65 | 53.98 | 47.07 | 40.71 |
| 11 | EdgeSpot T4 | MFCC | GE2E | 69.31 | 67.35 | 52.66 | 48.49 | 39.38 |
| 12 | EdgeSpot T4 | MFCC | SCAF+GE2E | 69.15 | 66.94 | 50.52 | 49.99 | 37.96 |
| 13 | EdgeSpot T4 | PCEN | Triplet | 70.76 | 68.81 | 60.18 | 43.41 | 44.36 |
| 14 | EdgeSpot T4 | PCEN | SCAF | 69.52 | 67.35 | 53.59 | 47.57 | 40.26 |
| 15 | **EdgeSpot T4** | **PCEN** | **GE2E** | **72.94** | **73.35** | **76.68** | **31.30** | **57.29** |
| 16 | EdgeSpot T4 | PCEN | SCAF+GE2E | 69.24 | 67.00 | 50.95 | 49.69 | 38.22 |

Ghi chú: param DSCNN-L ≈ 412,900; EdgeSpotFull T4 ≈ 130,598.

Nhận xét (mọi số đều có trên bảng):

- GE2E là loss mạnh nhất ở cả hai backbone (hàng 7 và 15 dẫn đầu nhóm).
- PCEN > MFCC khi đi với GE2E: DSCNN GE2E PCEN(76.67) vs MFCC(72.30); EdgeSpot GE2E PCEN(72.94) vs MFCC(69.31).
- GE2E > Triplet: DSCNN PCEN GE2E(76.67) vs Triplet(72.24); F1 GE2E(67.68) vs Triplet(54.98).
- SCAF và SCAF+GE2E sụt mạnh ở budget ngắn + nhiều lớp (AUC ~50, F1 ~38), thấp hơn GE2E rõ.

Quyết định shortlist: giữ `DSCNN-L + PCEN + GE2E` (hàng 7) và `EdgeSpotFull T4 + PCEN + GE2E` (hàng 15) để train dài hơn.

## 4. Bước 3 — Nâng hạn mức dữ liệu (cap) và đo bão hòa

**Thiết kế thí nghiệm (quan trọng).** Ở bước này frontend và loss được CỐ ĐỊNH, không biến thiên: **frontend = PCEN, loss = GE2E**. Lý do: phase-1 (mục 3, 16 tổ hợp) đã kết luận PCEN + GE2E là tổ hợp tốt nhất ở cả hai backbone, nên bước này chỉ giữ đúng 2 pipeline đã shortlist (`DSCNN-L + PCEN + GE2E` và `EdgeSpotFull T4 + PCEN + GE2E`) rồi biến thiên 2 trục còn lại: **cap (lượng dữ liệu)** và **backbone**. Việc biến thiên LOSS ở quy mô lớn được tách sang mục 6 (KD vs GE2E); việc biến thiên FRONTEND/LOSS đầy đủ đã làm ở mục 3.

Hai pipeline cố định, tăng dần số clip/từ (cap) trên Full MSWC (38,150 từ), train 20–25 epoch, GSC test100:

| Cap (clip/từ) | ~Số clip | DSCNN-L+PCEN+GE2E ACC@5%FAR | AUC | EER | F1 | EdgeSpot T4+PCEN+GE2E ACC@5%FAR | AUC | EER | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cap20 | ~0.53M | 86.05 | 91.57 | 16.25 | 75.90 | 83.06 | 87.22 | 20.40 | 70.46 |
| cap50 | ~0.94M | 84.68 | 90.45 | 17.42 | 74.34 | 82.24 | 87.74 | 20.19 | 70.73 |
| cap220 | ~2.05M | 88.23 | 93.87 | 12.78 | 80.67 | 86.03 | 91.31 | 16.47 | 75.61 |
| cap620 | ~2.99M | 88.56 | 94.04 | 12.53 | 81.02 | 86.01 | 91.34 | 16.64 | 75.38 |

Ghi chú: cả 4 hàng × 2 nhóm cột đều là **PCEN + GE2E**; chỉ khác backbone (DSCNN-L vs EdgeSpotFull T4) và cap.

Suy luận bão hòa:

- Bước nhảy lớn: cap50 → cap220 (DSCNN +3.55pp ACC@5%FAR; EdgeSpot +3.79pp).
- Sau đó phẳng: cap220 → cap620 (DSCNN chỉ +0.33pp; EdgeSpot −0.02pp) dù cap620 train nhiều hơn (1000 vs 800 episode) và nhiều clip hơn ~1M.
- Kết luận: accuracy bão hòa quanh ~2M clip. Bốn nguyên nhân: (1) ngân sách episodic cố định (1000 ep × 30×10 = 300k mẫu/epoch bất kể bể to cỡ nào), (2) tăng cap chỉ thêm clip của cùng các từ chứ không thêm từ/người nói mới, (3) trần chuyển giao MSWC→GSC (đánh giá trên 35 từ cố định), (4) trần dung lượng mô hình.
- Hệ quả: muốn vượt ~88% phải đổi loss/phương pháp (xem mục 6 — KD), KHÔNG phải thêm dữ liệu.

(cap20/cap50 chạy trên server; cap220/cap620 chạy trên Colab A100 — xem mục 7, 8.)

## 5. Bước 4 — Top500 (từ vựng hẹp 500 từ phổ biến)

GSC test100:

| Mô hình | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| EdgeSpotFull T4 + SCAF+GE2E (epoch13, legacy) | 85.62 | 88.79 | 95.34 | 11.51 | 82.45 | Checkpoint cũ, tái lập được; chất lượng tốt nhất ở dữ liệu lớn |
| DSCNN-L + PCEN + GE2E (recheck 20ep) | 81.55 | 86.56 | 93.17 | 14.00 | 78.97 | Train mới |
| EdgeSpotFull T4 + SCAF+GE2E (train-new 20ep) | 83.50 | 86.18 | 92.73 | 15.11 | 77.45 | Train mới ngắn, thua epoch13 |
| DSCNN-L + PCEN + SCAF+GE2E (e20_ep200, server 07-06) | 84.32 | 87.59 | 92.79 | 14.12 | 78.82 | 20 epoch × 200 episode, best@19; KW-ACC 90.24% |

Suy luận: ở từ vựng hẹp (Microset 31 từ, Top500 500 từ), SCAF+GE2E cho AUC ~95 — rất tốt. Khi train đủ lâu (epoch13), EdgeSpot+SCAF+GE2E vừa nhỏ vừa chính xác nhất. Đây là cơ sở chọn kiến trúc thesis.

## 5b. Vì sao các lần train sau có vẻ yếu hơn epoch13 (88.79% → ~86%)?

Đây KHÔNG phải do sai phương pháp. Khi so 88.79% với ~86%, ta đang so giữa các điều kiện train khác nhau. Bảng so sánh các lần (EdgeSpotFull T4 + SCAF+GE2E, GSC test100):

| Lần chạy | Dữ liệu (số từ) | episode/epoch | epoch (best) | max-per-word | ACC@5%FAR | AUC | EER | F1 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| Top500 epoch13 (run gốc) | Top500 (500 từ) | 300 | 13 (KH 25) | 0 = tất cả clip | 88.79 | 95.34 | 11.51 | 82.45 |
| Top500 recheck EdgeSpot (`e20_ep200`) | Top500 (500 từ) | 200 | trong 20 | full profile | 86.18 | 92.73 | 15.11 | 77.45 |
| Top500 DSCNN SCAF+GE2E (`e20_ep200`) | Top500 (500 từ) | 200 | trong 20 | full profile | 87.59 | 92.79 | 14.12 | 78.82 |
| Full MSWC cap220 GE2E | Full MSWC (38,150 từ) | 800 | trong 15 | cap 220 | 86.03 | 91.31 | 16.47 | 75.61 |
| Full MSWC cap620 GE2E | Full MSWC (38,150 từ) | 1000 | trong 20 | cap 620 | 86.01 | 91.34 | 16.64 | 75.38 |

(Config epoch13 lấy từ lệnh train gốc: `--epochs 25 --episodes 300 --max-per-word 0`; recheck lấy từ `docs/current_agent_state.md`, run ID `top500_full_recheck_e20_ep200`, `episodes=200`.)

Hai nguyên nhân chính (đọc trực tiếp từ bảng):

1. **Khác bộ dữ liệu / độ khó nhiệm vụ.** epoch13 (88.79) train trên Top500 = chỉ 500 từ phổ biến (dễ). Các con số ~86% của cap220/cap620 train trên Full MSWC = 38,150 từ (khó hơn nhiều, đa dạng người nói/từ). So 88.79 (500 từ) với 86 (38k từ) là so hai nhiệm vụ khác nhau.

2. **Khác recipe huấn luyện trên cùng Top500.** epoch13 dùng 300 episode/epoch và `max-per-word 0` (lấy toàn bộ clip), còn recheck chỉ 200 episode/epoch. episode/epoch ít hơn không chỉ giảm số bước cập nhật mỗi epoch mà còn làm **nhịp LR scheduler khác** (scheduler step theo epoch) → quỹ đạo tối ưu khác và đạt đỉnh thấp hơn 2.61 điểm. Đây là khác biệt recipe/ngân sách, không phải sai phương pháp.

Kết luận: không phải phương pháp kém đi. Muốn tái lập/ vượt 88.79% trên Top500, cần train SCAF+GE2E đúng recipe run gốc (300 episode/epoch, max-per-word 0, đủ epoch), và phải so cùng bộ dữ liệu (cùng Top500, không trộn với Full MSWC 38k từ).

## 6. Bước 5 — Knowledge Distillation (KD)

Teacher: Wav2Vec2 đóng băng + head 64-D train bằng Sub-center ArcFace (`scripts/train_teacher_head.py`). Student: EdgeSpotFull T4, loss `kd_scaf`.

Bảng đầy đủ KD vs GE2E (cùng EdgeSpotFull T4, cùng cap, GSC test100). Mọi số dùng để nhận xét đều có trên bảng:

| Cap | Loss | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 | KW-ACC |
|---|---|---:|---:|---:|---:|---:|---:|
| cap50 | GE2E | 77.14 | 82.24 | 87.74 | 20.19 | 70.73 | 83.49 |
| cap50 | KD (kd_scaf) | 80.74 | 85.82 | 91.19 | 15.41 | 77.04 | 87.95 |
| cap220 | GE2E | - | 86.03 | 91.31 | 16.47 | 75.61 | 88.29 |
| cap220 | KD (kd_scaf) | 80.82 | 85.90 | 91.17 | 15.36 | 77.10 | 87.44 |

(GE2E cap220 chỉ có ACC@5%FAR cuối, chưa eval lại @1%FAR — ô để `-`.)

Nhận xét (tất cả số đã có trong bảng trên):

- Ở dữ liệu ít (cap50): KD vượt GE2E ở mọi chỉ số — ACC@1%FAR 80.74 vs 77.14 (+3.60), ACC@5%FAR 85.82 vs 82.24 (+3.58), AUC 91.19 vs 87.74 (+3.45), EER 15.41 vs 20.19 (tốt hơn 4.78), F1 77.04 vs 70.73 (+6.31).
- KD cap50 (ACC@5%FAR 85.82) ≈ GE2E cap220 (86.03): KD đạt cùng độ chính xác với ~2× ít dữ liệu hơn.
- Ở cap220: KD ≈ GE2E về ACC@5%FAR (85.90 vs 86.03), nhưng KD tốt hơn về EER (15.36 vs 16.47) và F1 (77.10 vs 75.61).
- KD bão hòa sớm: cap50 → cap220 chỉ 85.82 → 85.90.

Kết luận KD: KD nâng EdgeSpot, đặc biệt ở chế độ ít dữ liệu, và cải thiện calibration (EER, F1) ở quy mô lớn. Đưa vào thesis như đóng góp.

## 7. Sự cố SCAF+GE2E ở quy mô lớn (37k lớp)

`EdgeSpotFull T4 + SCAF+GE2E` trên Full MSWC cap220 (37,387 lớp) với `scaf-weight=1.0` bị **sụp (collapse)**: từ epoch 2 trở đi Val AUC=0.5000, ge2e_acc=0.033 (ngẫu nhiên), eval test100 AUC=0.5, F1=0, threshold toàn 0.0. Nguyên nhân: SCAF (scale 30) với 37k lớp + trọng số 1.0 tạo gradient quá lớn phá vỡ embedding.

Đối chiếu: SCAF+GE2E chạy tốt ở Microset (31 lớp), Top500 (450–500 lớp) vì ít lớp. KD dùng `scaf-weight=5e-5` (đúng theo paper) nên không sụp.

Kết luận: ở 37k lớp, nếu muốn dùng SCAF phải hạ `scaf-weight` xuống ~5e-5; nếu không thì dùng GE2E hoặc KD. SCAF+GE2E cap220 collapse KHÔNG được báo cáo làm kết quả.

## 8. Trạng thái Colab — các lần chạy và nguồn số liệu

| Run id / mốc | Dữ liệu | Nội dung | Số liệu lấy từ |
|---|---|---|---|
| Microset (đã khóa) | 31 từ | DSCNN Triplet, EdgeSpot SCAF, EdgeSpot SCAF+GE2E | `reports/microset/` |
| Top500 full (cũ) | 500 từ | EdgeSpot SCAF+GE2E epoch13 (epoch25 chỉ log, mất checkpoint) | log Colab cũ + server reval |
| Full-WAV all-clips (06-04) | full | THẤT BẠI: tràn đĩa /content 236GB khi convert WAV | log Colab |
| `colab_mswc_heavy_flac_target6000000_20260604_171246` | cap220 FLAC (~2.05M) | DSCNN+EdgeSpot GE2E | `reports/colab_mswc_cap220_flac/summary_vi.md` |
| `colab_mswc_heavy_flac_target3000000_20260605_175317` | cap620 FLAC (~2.99M) | DSCNN+EdgeSpot GE2E | log Colab (dòng `Open-set ACC` ở khối cuối) |
| `kd_cap50_20260607` | cap50 FLAC | teacher head + precompute + EdgeSpot KD | log Colab (khối `FRR@5.0%FAR`/`FRR@1.0%FAR`) |
| `kd_cap220_20260607` | cap220 FLAC | EdgeSpot KD | log Colab (06:11 @5%, 06:17 @1%) |
| `cap220_scafge2e_20260607` | cap220 FLAC | EdgeSpot SCAF+GE2E → collapse (bỏ) | log Colab (AUC 0.5) |

Lưu ý nguồn số: trong log Colab, số ACC@5%FAR/ACC@1%FAR nằm ở dòng `Open-set ACC` trong khối in cuối mỗi lượt eval (xem mục 0).

## 9. Trạng thái Server (ict6) — sessions và số lần chết

Server ict6 (Tesla K80, CUDA 10.2, PyTorch 1.12). Các tmux session đã dùng:

- `kws_matrix12` (smoke 12-combo)
- `kws_matrix12_phase1_wait` / `kws_matrix_phase1_resume` (phase-1 12-combo)
- `kws_shortlist_manifest20` (cap20 shortlist)
- `kws_manifest50` / `kws_manifest50_fixed` / `kws_dscnn_max50_recovery` (cap50)
- `kws_top500_recheck` (Top500 recheck 3 nhánh)
- `kws_top500_edgespot_resume` (resume EdgeSpot Top500)
- `kws_top500_edgespot_eval1far` (eval @1%FAR)
- `kws_top500_dscnn_scafge2e` (hoàn thành 07-06-2026: DSCNN SCAF+GE2E Top500, test100 @1% và @5%FAR)

Các lần hỏng/chết đã gặp (ít nhất 5 loại):

1. ict6 mất kết nối nhiều lần: `Connection timed out during banner exchange` (kéo dài 06-02 → 06-04, khoảng ~2 ngày không vào được node GPU).
2. Top500 EdgeSpot train-new bị treo ở `Epoch 11/20` (06-02 21:04) khi server kẹt.
3. `OSError: [Errno 12] Cannot allocate memory` khi DataLoader fork với `workers=8` → train `failed_rc_1` (06-04 08:48). Khắc phục: chạy với `--num-workers 0`.
4. manifest50 DSCNN lỗi đọc audio NFS (`soundfile LibsndfileError` trên file `.opus`) ở epoch 14 → phải viết script recovery resume từ `latest.pt`.
5. PyTorch 1.12 không hỗ trợ `torch.load(weights_only=False)` → phải patch `load_checkpoint` để fallback; relaunch phase-1 resume.
6. ict14 vào được nhưng không có GPU/CUDA env phù hợp → không dùng để train được.

Số liệu server lấy từ: `reports/server_far_metrics/server_far_metrics.csv` và log `/storage/<user>/an_kws/logs/*.tsv` / `*.log`.

## 10. Bảng tổng hợp cuối cùng (GSC test100)

| Dữ liệu (từ) | Mô hình | Loss | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 | Tham số |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Microset (31) | EdgeSpot T4 | SCAF+GE2E (KHÓA) | 84.61 | 86.12 | 95.61 | 11.54 | 82.41 | 131k |
| Top500 (500) | EdgeSpot T4 | SCAF+GE2E (ep13) | 85.62 | 88.79 | 95.34 | 11.51 | 82.45 | 131k |
| Top500 (500) | DSCNN-L | GE2E | 81.55 | 86.56 | 93.17 | 14.00 | 78.97 | 413k |
| Top500 (500) | DSCNN-L | SCAF+GE2E (e20) | 84.32 | 87.59 | 92.79 | 14.12 | 78.82 | 413k |
| Full cap20 (38k) | DSCNN-L | GE2E | 82.10 | 86.05 | 91.57 | 16.25 | 75.90 | 413k |
| Full cap220 (38k) | DSCNN-L | GE2E | - | 88.23 | 93.87 | 12.78 | 80.67 | 413k |
| Full cap620 (38k) | DSCNN-L | GE2E | - | 88.56 | 94.04 | 12.53 | 81.02 | 413k |
| Full cap20 (38k) | EdgeSpot T4 | GE2E | 79.58 | 83.06 | 87.22 | 20.40 | 70.46 | 131k |
| Full cap220 (38k) | EdgeSpot T4 | GE2E | - | 86.03 | 91.31 | 16.47 | 75.61 | 131k |
| Full cap620 (38k) | EdgeSpot T4 | GE2E | - | 86.01 | 91.34 | 16.64 | 75.38 | 131k |
| Full cap50 (38k) | EdgeSpot T4 | KD (kd_scaf) | 80.74 | 85.82 | 91.19 | 15.41 | 77.04 | 131k |
| Full cap220 (38k) | EdgeSpot T4 | KD (kd_scaf) | 80.82 | 85.90 | 91.17 | 15.36 | 77.10 | 131k |
| Full cap220 (38k) | EdgeSpot T4 | SCAF+GE2E | collapse | collapse | 50.0 | 50.0 | 0.0 | 131k |

## 11. Kết luận tổng

1. Kiến trúc: chọn `EdgeSpotFull T4 + PCEN` cho thiết bị (131k tham số), `DSCNN-L` cho accuracy tham chiếu (413k).
2. Loss: GE2E là nền tốt nhất ở quy mô lớn; SCAF+GE2E tốt nhất ở từ vựng hẹp (Microset/Top500) nhưng sụp ở 37k lớp với trọng số mặc định; KD tốt nhất cho mô hình nhỏ ở chế độ ít dữ liệu.
3. Dữ liệu: bão hòa quanh ~2M clip; tăng thêm không cải thiện. Giải pháp tiếp theo là phương pháp (KD/SCAF tuned), không phải thêm dữ liệu.
4. Đóng góp nghiên cứu: KD cho EdgeSpot đạt cùng accuracy với ~2× ít dữ liệu hơn GE2E, và cải thiện EER/F1 ở quy mô lớn.

## 12. Còn thiếu / nên làm

- ACC@1%FAR cuối cho cap220/cap620 GE2E (mới có @5%FAR): cần eval lại best.pt với `--target-far 0.01`.
- (Tuỳ chọn) SCAF+GE2E cap220 với `--scaf-weight 5e-5` nếu muốn số hợp lệ cho bảng ablation.
- Copy artifact cap220/cap620/KD từ Drive về `reports/` để khóa bằng chứng local.
