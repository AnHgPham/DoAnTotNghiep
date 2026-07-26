# Báo cáo tối ưu đồ án KWS - 2026-06-01

## 1. Mục tiêu

Tối ưu hướng nghiên cứu và vận hành thí nghiệm cho đồ án few-shot open-set keyword spotting. Mục tiêu không phải chạy lại toàn bộ 16 pipeline một cách dàn trải, mà là dùng kết quả đã có để chọn đúng nhánh cần train/evaluate tiếp, giảm rủi ro server, và giữ claim nghiên cứu có bằng chứng.

## 2. Kết luận điều phối sáu vai trò

| Vai trò | Kết luận chính | Hành động tương ứng |
|---|---|---|
| Main/Supervisor | Không mở thêm broad matrix; ưu tiên shortlist có bằng chứng mạnh. | Chọn 2 pipeline PCEN+GE2E để train/evaluate tiếp. |
| Codebase Engineer | `max50` trước đó bị hỏng ở bước manifest; cần đọc từ extracted clips thay vì quét archive lớn. | Sửa manifest builder, thêm test local/remote. |
| ML/Data Engineer | Full uncapped MSWC quá tốn thời gian; `max50` là bước tăng dữ liệu hợp lý sau manifest20. | Train full vocabulary capped 50 clips/word, không train toàn bộ uncapped. |
| Evaluation Scientist | Bằng chứng mạnh nhất hiện tại là GSC-test100 của shortlist manifest20. | Dùng test100 mean/std để báo cáo, không dùng phase-1 dev làm kết luận cuối. |
| UI/Docs Engineer | Thesis/email cần nói rõ khác nhau giữa Microset, Top500, Full MSWC manifest20/max50. | Viết báo cáo theo bảng: architecture, feature, loss, metric. |
| Ops/QA Engineer | Server có dữ liệu `.opus`, không phải `.wav`; GPU 4 idle và phù hợp chạy job. | Sửa hỗ trợ `.opus`, launch tmux job `kws_manifest50_fixed`. |

## 3. Việc đã làm

### 3.1. Chẩn đoán lỗi `max50`

Lần chạy `max50` cũ dừng ở manifest construction, chưa tạo được:

- `data/mswc_en/splits/train_files_max50.json`
- `data/mswc_en/splits/val_files_max50.json`
- `data/mswc_en/splits/file_manifest_summary_max50.json`

Nguyên nhân kỹ thuật:

- script cũ có xu hướng quét `data/mswc_en/en.tar.gz`, rất nặng trên NFS;
- khi chuyển sang extracted clips, code ban đầu chỉ lọc `.wav`;
- dữ liệu MSWC trên ict6 thực tế là `.opus`, ví dụ `data/mswc_en/clips/heron/common_voice_en_19732438.opus`;
- nếu không sửa, manifest có nguy cơ rỗng hoặc rất chậm.

### 3.2. Sửa code manifest

File đã sửa:

- `data/build_mswc_file_splits.py`
- `tests/test_build_mswc_file_splits.py`
- `server/run_full_mswc_shortlist_manifest50.sh`
- `server/launch_shortlist_manifest50_fixed.sh`

Thay đổi chính:

- thêm chế độ `--source clips` để build manifest từ `data/mswc_en/clips`;
- hỗ trợ audio suffix `.wav`, `.opus`, `.flac`, `.mp3`;
- với `max_per_word=50`, dừng đọc mỗi word folder sau khi đủ 50 audio file để tránh quét/sort toàn bộ folder lớn;
- launcher mới dùng tmux session `kws_manifest50_fixed`;
- runner dùng `RUN_ID=full_mswc_shortlist_manifest50_clips_e20_ep200` để tách log khỏi lần hỏng cũ.

### 3.3. Kiểm chứng

Local:

```bash
python -m py_compile data/build_mswc_file_splits.py
python -m pytest tests/test_build_mswc_file_splits.py -q
```

Kết quả local: `3 passed`.

Server ict6:

```bash
python -m py_compile data/build_mswc_file_splits.py
python -m pytest tests/test_build_mswc_file_splits.py -q
bash -n server/run_full_mswc_shortlist_manifest50.sh server/launch_shortlist_manifest50_fixed.sh
```

Kết quả server: `3 passed`, bash syntax OK.

### 3.4. Job đã launch

Trạng thái đã quan sát trên ict6:

- tmux session: `kws_manifest50_fixed`
- runner: `/storage/<user>/an_kws/DoAnTotNghiep/server/run_full_mswc_shortlist_manifest50.sh`
- bootstrap log: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_bootstrap.log`
- wait log: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_wait_gpu.log`
- GPU: `4`
- manifest mode: `source=clips`
- manifest cap: `max_per_word=50`

Job đang ở bước:

```text
Building MSWC file manifests from clips
Targets: 37387 train words, 763 val words, max_per_word=50, source=clips
```

## 4. Bằng chứng hiện có để báo cáo ngay

Kết quả chắc nhất hiện tại là shortlist manifest20 đã train/evaluate xong trên GSC-dev30 và GSC-test100.

| Pipeline | Split | Runs | ACC@1%FAR | ACC@5%FAR | AUC | EER | KW-ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | test100 | 100 | 82.10 +/- 0.87 | 86.05 +/- 0.66 | 91.57 +/- 0.58 | 16.25 +/- 0.86 | 88.90 +/- 1.25 | 75.90 +/- 1.16 |
| EdgeSpotFull T4 + PCEN + GE2E | test100 | 100 | 79.58 +/- 0.91 | 83.06 +/- 0.82 | 87.22 +/- 0.75 | 20.40 +/- 1.01 | 83.01 +/- 1.49 | 70.46 +/- 1.30 |

Diễn giải:

- nếu ưu tiên accuracy: `DSCNN-L + PCEN + GE2E` đang mạnh hơn;
- nếu ưu tiên edge/compact: `EdgeSpotFull T4 + PCEN + GE2E` vẫn có giá trị vì nhỏ hơn đáng kể;
- EdgeSpotFull T4 kém DSCNN-L khoảng `2.52 pp` ACC@1%FAR và `5.44 pp` F1 trên test100;
- kết quả này mạnh hơn phase-1 vì dùng `test100` và mean/std.

## 5. Vì sao không train full uncapped MSWC ngay

Full MSWC English có metadata khoảng `6.62M` clips và `38,174` words. Train uncapped sẽ rất tốn thời gian, dễ nghẽn I/O, và chưa chắc đem lại insight tốt hơn trong tuần này.

Chiến lược hợp lý hơn:

1. dùng phase-1 manifest20 để chọn pipeline;
2. train/evaluate shortlist với manifest20 test100 để có bằng chứng ổn;
3. tăng từ `max20` lên `max50` cho đúng 2 pipeline tốt nhất;
4. chỉ nếu `max50` thật sự cải thiện, mới cân nhắc tăng cap hoặc train lâu hơn.

Như vậy `max50` là bước nghiên cứu kiểm soát biến số: cùng full vocabulary, chỉ tăng số file/word từ 20 lên 50.

## 6. Kỳ vọng kết quả

Pipeline đang chạy tiếp:

| Pipeline | Mục đích |
|---|---|
| DSCNN-L + PCEN + GE2E | kiểm tra ceiling accuracy khi tăng dữ liệu/word |
| EdgeSpotFull T4 + PCEN + GE2E | kiểm tra compact model có thu hẹp gap với DSCNN không |

Kỳ vọng hợp lý:

- ACC@1%FAR có thể tăng nhẹ hoặc ổn định hơn, khoảng `+0` đến `+2/3 pp`; không nên hứa chắc tăng;
- EER/FRR có thể giảm nếu thêm samples/word giúp prototype ổn định hơn;
- nếu EdgeSpot cải thiện nhiều hơn DSCNN, luận văn có thể nhấn mạnh trade-off compactness vs accuracy;
- nếu DSCNN vẫn dẫn rõ, luận văn nên tách kết luận thành hai hướng: best accuracy và edge candidate.

## 7. Ước lượng thời gian

Từ lúc launch lại job fixed lúc khoảng `2026-06-01 23:31 ICT`:

| Giai đoạn | Ước lượng |
|---|---:|
| Build manifest max50 từ clips | 45-90 phút, phụ thuộc NFS |
| Train/evaluate DSCNN-L + PCEN + GE2E | 5-7 giờ |
| Train/evaluate EdgeSpotFull T4 + PCEN + GE2E | 7-11 giờ |
| Tổng | khoảng 14-22 giờ |

Mốc có thể kiểm tra:

- Sau manifest: xuất hiện `train_files_max50.json`, `val_files_max50.json`, `file_manifest_summary_max50.json`.
- Khi train bắt đầu: log có dòng `Starting experiment: dscnn_pcen_ge2e...`.
- Khi xong hoàn toàn: TSV có 2 dòng `ok/ok/ok` tại `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv`.

## 8. Nội dung có thể báo cáo cho thầy hiện tại

Có thể báo cáo:

- Em đã hoàn thành phase shortlist trên Full MSWC English manifest20 với GSC-test100.
- `PCEN + GE2E` là hướng ổn định nhất trong cả DSCNN-L và EdgeSpotFull T4.
- `DSCNN-L + PCEN + GE2E` đạt `82.10 +/- 0.87%` ACC@1%FAR trên test100.
- `EdgeSpotFull T4 + PCEN + GE2E` đạt `79.58 +/- 0.91%` ACC@1%FAR trên test100, thấp hơn nhưng compact hơn.
- Em đang chạy follow-up `max50` cho đúng 2 pipeline này để kiểm tra ảnh hưởng của tăng số samples/word.

Không nên báo cáo:

- Không nói `max50` đã có kết quả cho đến khi TSV/JSON tồn tại.
- Không nói đã train full uncapped 6.62M clips.
- Không dùng phase-1 dev result thay cho test100.
- Không dùng demo UI sampled evaluation thay cho `gsc_edgespot_exact test100`.

## 9. Lệnh kiểm tra tiếp

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux ls | grep kws_manifest50_fixed
tail -n 80 /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_bootstrap.log
ls -lh /storage/<user>/an_kws/DoAnTotNghiep/data/mswc_en/splits/*max50*
cat /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv
```

## 10. Quyết định sau khi có kết quả `max50`

Nếu `DSCNN-L + PCEN + GE2E` vẫn tốt nhất:

- dùng DSCNN-L làm best-accuracy baseline/final candidate;
- dùng EdgeSpotFull T4 để thảo luận trade-off model size vs accuracy.

Nếu `EdgeSpotFull T4 + PCEN + GE2E` thu hẹp gap:

- ưu tiên EdgeSpotFull T4 trong thesis vì có tính edge/device rõ hơn;
- có thể thêm một run dài hơn hoặc tuning nhẹ cho EdgeSpot.

Nếu `max50` không cải thiện:

- giữ manifest20 test100 là evidence chính cho Full MSWC;
- không tăng dataset cap nữa, chuyển sang viết thesis và dọn báo cáo/figure.
