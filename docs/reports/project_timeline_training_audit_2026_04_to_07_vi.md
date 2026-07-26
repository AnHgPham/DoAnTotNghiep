# Audit tiến độ và thực nghiệm từ tháng 4 đến tháng 7/2026

**Ngày đối chiếu:** 13/07/2026  
**Phạm vi:** mã nguồn, dữ liệu, training, evaluation, ICTLab/ict6, Colab, demo,
thesis, slide và script thuyết trình.

## 1. Cách đọc mức bằng chứng

- **A - đã khóa:** có TSV/JSON/checkpoint/artifact cục bộ hoặc log gốc đã copy từ
  server.
- **B - có ghi nhận:** có báo cáo hoặc log lịch sử, nhưng thiếu một phần artifact
  gốc ở local.
- **C - kế hoạch/không hoàn tất:** mới chuẩn bị, bị dừng, bị lỗi hoặc chỉ đánh giá
  checkpoint có sẵn. Không được tính như một lần training hoàn chỉnh.

## 2. Kết luận kiểm tra nhanh

1. Có ghi chú rõ giai đoạn chạy trên ICTLab, cụ thể là node `ict6`, không phải chỉ
   chạy Colab.
2. Đường dẫn làm việc trên server là
   `/storage/<user>/an_kws/DoAnTotNghiep`; môi trường chính là Python 3.9,
   PyTorch 1.12.1+cu102 trên Tesla K80 và thường giới hạn `CUDA_VISIBLE_DEVICES=4`.
3. ICTLab được dùng để tải/chuẩn bị Full MSWC, smoke test, screening 16 pipeline,
   shortlist cap20/cap50, Top500 recheck và một số ablation.
4. Matrix `40 epoch x 150 episode` khởi chạy trên ict6 ngày 11/06 **không hoàn
   thành 16/16**. Audit TSV ngày 13/07 cho thấy chỉ 1 cấu hình hoàn tất, 15 cấu
   hình `failed_rc_1`, chủ yếu liên quan hết dung lượng và một run bị dừng giữa
   chừng. Cấu hình duy nhất hoàn tất là `DSCNN + PCEN + SCAF`; nhánh
   `DSCNN + PCEN + GE2E` đi đến epoch30/40 rồi dừng. Matrix 16 pipeline hoàn
   chỉnh dùng trong thesis là bản **Colab A100 cap620**.
5. Kết quả cuối dùng cho slide/thesis là hai run development Colab:
   `DSCNN-L + PCEN + GE2E = 86.36% ACC@1%FAR` và
   `EdgeSpotFull T4 + PCEN + GE2E = 82.87% ACC@1%FAR` trên GSC test100.

## 3. Timeline theo tháng

### Tháng 4/2026 - Khởi tạo nền tảng

**Mức bằng chứng: A/B.**

- Ngày 03-05/04: tạo cấu trúc `src/`, `configs/`, `data/`, `tests/` và tài liệu
  proposal/outline.
- Hoàn thiện MFCC ban đầu: audio 16 kHz, 1 giây -> `(1, 47, 10)`.
- Hoàn thiện DSCNN-L ban đầu: 5 depthwise-separable blocks, embedding 276 chiều,
  chuẩn hóa L2.
- Viết script tải GSC v2 và test cho MFCC/DSCNN.
- Lập kế hoạch tiếp theo cho episodic sampler, Triplet Loss, augmentation,
  OpenNCM và training script.
- Chưa tìm thấy log training matrix có timestamp đầy đủ trong tháng 4, nên không
  gán các run training về tháng này nếu chỉ dựa trên tên checkpoint về sau.

Lưu ý ngày: `docs/weekly_report_week1.md` ghi `Sunday, [4/5/2026]`. Đối chiếu
thứ trong tuần và timestamp cho thấy đây là **05/04/2026** theo định dạng
tháng/ngày, không phải 04/05/2026.

### Tháng 5/2026 - Baseline, chọn phương pháp và chuyển lên ICTLab

**Đầu tháng 5 - Baseline và open-set:**

- So sánh OpenNCM, OpenMAX và Energy; OpenNCM vẫn là baseline chính.
- Checkpoint DSCNN + MFCC + Triplet `best_v2_margin1.0_colab.pt` cải thiện GSC
  fixed AUC lên khoảng 0.967.
- Phase 2 hard-pair mining làm giảm generalization trên GSC, nên được giữ như
  negative result; `best_v2` vẫn là checkpoint baseline.
- Top500 DSCNN + MFCC + Triplet đạt MSWC validation tốt nhưng cross-dataset GSC
  còn yếu (`KW-ACC=69.8%` ở fixed k=5), từ đó xác định vấn đề domain shift và
  giới hạn của Triplet.

**Giữa tháng 5 - Nâng cấp kiến trúc và Microset:**

- Thêm EdgeSpotFull T4, PCEN, SCAF, GE2E/hybrid và protocol
  `gsc_edgespot_exact`.
- Chạy 3 cấu hình Microset:
  1. DSCNN-L + MFCC + Triplet.
  2. EdgeSpotFull T4 + PCEN + SCAF.
  3. EdgeSpotFull T4 + PCEN + SCAF+GE2E.
- Microset có 96,099 file theo official file-level split.
- Cấu hình thứ ba đạt GSC test100: ACC@1%FAR 84.61%, ACC@5%FAR 86.12%,
  AUC 95.61%, EER 11.54%, Keyword ACC 77.66%, F1 82.41%.

**Top500 trên Colab:**

- Run đầu hoàn tất đến epoch25 và log checkpoint-selection GSC-dev 3 runs ghi
  ACC@1%FAR 87.61%, ACC@5%FAR 89.20%. Checkpoint epoch25 bị mất trước khi
  package, nên đây là mức B, không phải artifact khóa.
- Run lại bị dừng ở epoch13 vì hết Colab compute units. Epoch13 là artifact local
  chắc chắn, dev30 đạt ACC@1%FAR 86.68% và ACC@5%FAR 88.87%.

**27-29/05 - Thiết lập ICTLab/ict6 và Full MSWC:**

- Luồng SSH: `<lab-gateway>:<port>` -> frontend -> `ssh ict6`.
- Sửa môi trường server: Python 3.9, CUDA 10.2, torchaudio CUDA thay vì ROCm,
  annotation tương thích Python 3.9 và bộ requirements riêng.
- GSC v2 được tải và giải nén trên server.
- Full MSWC English được tải từ mirror, archive khoảng 34.8 GB.
- Metadata chuẩn ghi 38,174 từ và 6,624,343 clip tiếng Anh.
- Split do dự án tạo theo word-disjoint: 37,387 train words, 763 val words,
  24 từ bị loại vì thiếu dữ liệu.
- Quá trình duyệt archive hoàn tất 6,662,520 entry ngày 29/05. Các phép train sau
  không quét toàn thư mục tự do mà dùng manifest khóa.
- Manifest cap20 dùng cho screening có 527,069 train files và 10,637 val files.

**29-31/05 - Bắt đầu training thực tế trên ICTLab:** xem bảng ở mục 4.

### Tháng 6/2026 - Mở rộng dữ liệu, matrix đầy đủ và khóa kết quả

**ICTLab/ict6:**

- Hoàn tất phase-1 16 pipeline cap20, sau đó shortlist PCEN+GE2E cap20 và cap50.
- Chạy lại Top500 với DSCNN+GE2E và EdgeSpot+SCAF+GE2E; EdgeSpot phải resume
  với `num_workers=0` sau lỗi RAM.
- Chạy thêm DSCNN + PCEN + SCAF+GE2E Top500 20 epoch x 200 episode ngày 07/06;
  test100 ACC@1%FAR 84.32%, ACC@5%FAR 87.59% theo báo cáo tổng hợp. Artifact
  JSON đánh giá có ở local, nhưng thiếu TSV training đầy đủ nên xếp mức B.
- Matrix fixed40 trên ict6 bị lỗi diện rộng; không được dùng làm matrix hoàn chỉnh.

**Colab A100 và Full MSWC FLAC:**

- Thử all-WAV Full MSWC ngày 04/06 thất bại trước training do tràn ổ `/content`.
- Chuyển sang FLAC và estimator chọn cap.
- Cap220: 2,049,717 file, 15 epoch x 800 episode, hoàn tất DSCNN+GE2E và
  EdgeSpot+GE2E. ACC@5%FAR lần lượt 88.23% và 86.03%.
- Cap620 heavy: khoảng 2.99 triệu file, 20 epoch x 1,000 episode, hoàn tất 2
  model GE2E. ACC@5%FAR lần lượt 88.56% và 86.01%.
- Chạy KD cap50/cap220 và một SCAF+GE2E cap220. KD cải thiện calibration cho
  EdgeSpot; SCAF+GE2E với trọng số 1.0 trên khoảng 37k lớp bị collapse
  (`AUC=0.5`, `F1=0`). Các mốc này được ghi trong master synthesis, nhưng một
  phần chỉ còn log/tổng hợp, nên không phải bảng chính cuối cùng.

**11-12/06 - Matrix cap620 hoàn chỉnh trên Colab:**

- 16 pipeline = 2 encoder x 2 frontend x 4 objective.
- Cùng profile cap620, 40 epoch x 150 episode, 30-way x 10-sample.
- Tất cả 16 cấu hình hoàn tất.
- End-to-end từ 11/06 15:45:17 đến 12/06 02:44:06: 10 giờ 58 phút 49 giây,
  gồm chuẩn bị dữ liệu, train, eval và sync artifact.
- Screening tốt nhất: DSCNN-L + PCEN + GE2E 82.34% ACC@1%FAR.
- Nhánh compact tốt nhất: EdgeSpotFull T4 + PCEN + GE2E 79.98% ACC@1%FAR.

**12/06 - Development run cuối:**

- 3 cấu hình, 60 epoch x 300 episode, 40-way x 10-sample.
- DSCNN + PCEN + GE2E.
- EdgeSpot + PCEN + Triplet hard.
- EdgeSpot + PCEN + GE2E.
- End-to-end từ 05:06:14 đến 12:05:35: 6 giờ 59 phút 21 giây, gồm khoảng
  3 giờ 32 phút chuẩn bị data.
- Kết quả khóa: DSCNN 86.36%, EdgeSpot GE2E 82.87%; nhánh Triplet hard collapse
  về khoảng 69.10% ACC@1%FAR.

**Thesis và tài liệu:**

- Khóa bảng test100, DET/heatmap, giải thích FAR/FRR/EER/AUC và giới hạn claim.
- Viết các bản thesis tiếng Việt/Anh, chương khoa học cap620 và báo cáo tuần.
- Giữ nguyên nguyên tắc: MSWC dùng train, GSC `gsc_edgespot_exact test100` là
  benchmark chính; không trộn số liệu Microset, Top500 và Full MSWC trong cùng
  ranking nếu không ghi rõ profile.

### Tháng 7/2026 - Production demo, slide, script và chuẩn bị bảo vệ

**Không có bằng chứng về một đợt retrain model mới trong tháng 7.** Hai
checkpoint composite-300 từ tháng 6 được dùng cho production/demo.

- Chọn hai profile mặc định:
  - DSCNN-L + PCEN + GE2E composite-300, 86.36%, 412,900 tham số.
  - EdgeSpotFull T4 + PCEN + GE2E composite-300, 82.87%, khoảng 130,598 tham số.
- Tối ưu demo: cache feature/enrollment, batching, vector hóa prototype,
  long-audio async, lock model, upload limits, CORS, health/warmup/timing và hai
  model profile rõ ràng.
- Benchmark CPU local DSCNN: median inference 29.73 ms, server total 33.56 ms,
  HTTP local 64.76 ms. Long audio đạt khoảng 8.51x real-time trong benchmark.
- Kiểm thử: 165 Python tests pass; UI typecheck/build pass; npm audit không có
  vulnerability; đã kiểm tra desktop/mobile.
- Hoàn thiện slide 17 trang, script khoảng 8 phút cho trình độ IELTS 5.5 và kế
  hoạch học source code trước ngày bảo vệ.

## 4. Audit chi tiết các lần chạy ICTLab/ict6

| Thời gian | Nhóm chạy | Cấu hình | Trạng thái | Thời lượng quan sát | Bằng chứng |
|---|---|---|---|---:|---|
| 29/05 20:23-21:09 | Smoke matrix | 12 cấu hình, 1 epoch x 20 episode | 12/12 OK | 46m24s | A, TSV |
| 30/05 03:05-19:23 | Phase-1 | 12 chính + 4 EdgeSpot-MFCC, 5 epoch x 150 episode | 16/16 OK | 16h17m38s wall | A, TSV/JSON |
| 31/05 10:34-22:14 | Shortlist cap20 | DSCNN/EdgeSpot + PCEN + GE2E, 20 x 200 | 2/2 OK | 11h40m34s | A, TSV/JSON |
| 02/06 00:26-11:59 | Shortlist cap50 | 2 model, 20 x 200 | 2 model hoàn tất; DSCNN lỗi epoch14 rồi resume | 11h32m34s wall | A, TSV/log/JSON |
| 02-05/06 | Top500 recheck | DSCNN+GE2E và EdgeSpot+SCAF+GE2E, 20 x 200 | 2/2 hoàn tất; EdgeSpot lỗi RAM rồi resume | nhiều phiên, có outage | A, TSV/log/JSON |
| 07/06 | Top500 ablation | DSCNN+PCEN+SCAF+GE2E, 20 x 200 | Hoàn tất theo report; thiếu TSV train local | chưa khóa | B, report + eval JSON |
| 11-12/06 | Fixed40 cap20 | 16 cấu hình, 40 x 150 | **DSCNN+PCEN+SCAF OK; 15 failed_rc_1** | cửa sổ 23h56m | A, TSV/log audit |

Các lỗi vận hành đã ghi nhận trên ict6:

- node mất kết nối/timed out trong một số khoảng;
- `Cannot allocate memory` với DataLoader workers=8;
- lỗi đọc OPUS/NFS ở cap50, phải resume;
- `No space left on device` trong matrix fixed40;
- PyTorch 1.12 cần fallback khi load checkpoint;
- ict14 không có môi trường GPU phù hợp, nên training chính vẫn ở ict6.

## 5. Số lần train: cách trả lời không gây hiểu nhầm

Không nên trả lời chỉ bằng một con số mà không định nghĩa. Cách đếm kiểm chứng
được từ các pha đã liệt kê:

- **Final controlled evidence:** 16 fixed cap620 + 3 development = **19 run**.
- **ICTLab hoàn tất:** 12 smoke + 16 phase-1 + 2 cap20 + 2 cap50 + 2 Top500
  recheck + 1 Top500 ablation + 1 nhánh fixed40 = **36 training job hoàn tất**.
- **Các pha Colab khác đã ghi nhận:** 3 Microset + 1 Top500 epoch25 + 2 cap220
  + 2 cap620 heavy + 2 KD + 1 SCAF+GE2E collapse = **11 job**.
- Tổng tối thiểu theo các pha trên: **66 training job kết thúc**, trong đó có
  12 smoke và ít nhất 1 run collapse. Bỏ smoke còn **54 job**.

Đây là **lower bound có cấu trúc**, không phải tổng tuyệt đối của mọi lệnh từng
chạy. Nó chưa cộng các baseline rất sớm `best_v1/best_v2`, teacher-head KD,
Phase-2 hard-pair, run song song, run bị dừng epoch13 và các lần eval-only, vì
log bắt đầu/kết thúc của chúng không đồng nhất. Không được dùng 66 như “66 thí
nghiệm khoa học độc lập”; nhiều job là smoke, retry, ablation hoặc profile khác.

## 6. Thời gian training: điều có thể khẳng định

- Không có một tổng GPU-hour chính xác cho toàn dự án vì Colab mất session,
  server có outage, nhiều run resume và một số log lịch sử thiếu timestamp.
- Pha cuối có timestamp đầy đủ:
  - fixed16 Colab end-to-end: 10h58m49s;
  - development3 Colab end-to-end: 6h59m21s;
  - cộng hai pha: **17h58m10s wall time**.
- Riêng ICTLab có các cửa sổ đo được trong bảng mục 4, nhưng không nên cộng thẳng
  thành GPU compute vì có chờ GPU, evaluation, outage và resume.

## 7. Câu trả lời ngắn khi hội đồng hỏi

> Dự án được phát triển từ tháng 4 đến tháng 7. Tháng 4 em xây baseline MFCC,
> DSCNN và dữ liệu GSC. Tháng 5 em đánh giá baseline, bổ sung EdgeSpot, PCEN,
> SCAF và GE2E, sau đó chọn hướng bằng Microset và mở rộng Top500. Từ cuối tháng
> 5 đến tháng 6, em chuyển sang ICTLab ict6 để tải Full MSWC và chạy smoke,
> screening 16 pipeline, cap20/cap50 và Top500 recheck. Sau đó em dùng Colab A100
> để hoàn tất matrix cap620 16 pipeline trong điều kiện đồng nhất và train sâu ba
> nhánh tốt nhất. Kết quả cuối là DSCNN+PCEN+GE2E 86.36% ACC@1%FAR và EdgeSpot
> T4+PCEN+GE2E 82.87% trên GSC test100. Tháng 7 em không retrain mà tập trung
> tối ưu demo, kiểm thử, thesis, slide và script bảo vệ.

## 8. Nguồn bằng chứng chính

- `docs/weekly_report_week1.md`
- `docs/inventory_review.md`
- `docs/phase2_negative_result.md`
- `docs/training_limitations_report_vi.md`
- `docs/tier1_research_upgrade.md`
- `docs/colab_microset_experiment_report_vi.md`
- `docs/reports/weekly_report_2026_05_18_22_vi.md`
- `docs/session_handoff_2026_05_29.md`
- `docs/reports/master_synthesis_2026_06_07_vi.md`
- `docs/reports/weekly_report_2026_06_10_14_vi.md`
- `docs/session_handoff_2026_07_11_production_demo.md`
- `reports/full_mswc_matrix_analysis/raw/*.tsv`
- `reports/full_mswc_shortlist_manifest20/raw/*.tsv`
- `reports/full_mswc_shortlist_manifest50/logs/*.tsv`
- `reports/server_metrics_raw/ict6_audit_20260713/*`
- `reports/colab_cap620_fixed_raw/colab_mswc_cap620_fixed16_20260611_154517.txt`
- `results/cap620_16_pipeline_metrics_long.csv`
- `results/logs_colab/colab_mswc_cap620_development_20260612_050614/stages.tsv`
