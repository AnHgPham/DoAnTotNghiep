# Inventory & Cleanup Report

> **Status:** DONE — đã clean và đánh nhãn lại theo plan đã duyệt.
> **Ngày:** 2026-05-08

---

## TL;DR — phát hiện chính

1. **Best checkpoint thực sự** là `epoch_30.pt` (val_auc = 0.9395), **không phải** `best.pt` cũ trên Colab (val_auc = 0.9250).
   Logic save best đã không cập nhật khi resume từ epoch 22.
   → Đã đổi tên `epoch_30.pt` thành `checkpoints/triplet/best_v2_margin1.0_colab.pt`.

2. **Eval với best_v2 đã được chạy sẵn** trên Colab — kết quả nằm trong `kws_checkpoints_results/results/`. Đã copy về `results/` (canonical) và move kết quả cũ sang `results/v1_margin0.5/`.

3. **Cải thiện AUC từ retrain (margin 0.5 → 1.0, 35 epoch, cosine restarts):**

| Protocol | best_v1 (cũ) | best_v2 (mới) | Δ AUC | FRR@5%FAR (mới) |
|---|---|---|---|---|
| GSC fixed k=5 | 0.9098 | **0.9671** | **+5.7%** | 0.2052 (giảm từ 0.379) |
| GSC fixed k=10 | 0.9229 | **0.9674** | **+4.5%** | 0.1768 |
| GSC random k=5 | 0.8751 | **0.9197** | **+4.5%** | 0.4948 |
| GSC random k=10 | 0.8819 | **0.9215** | **+4.0%** | 0.5012 |

→ Bài toán đã đạt mức tốt rất gần với `calibrated/gsc_fixed_results.json` (AUC = 0.9515) **chỉ nhờ retrain encoder**, chưa cần per-class threshold.

---

## Layout sau cleanup

### `checkpoints/`

```
checkpoints/
└── triplet/
    ├── best_v1_margin0.5_local.pt        (4.8 MB, epoch=15, baseline cũ)
    ├── best_v2_margin1.0_colab.pt        (4.8 MB, epoch=29, val_auc=0.9395) ← CANONICAL
    ├── runs/                             (TB log run cũ — 1 file, ~1 KB)
    └── runs_v2/                          (TB log run mới — 6 file Colab)
```

### `results/`

```
results/
├── gsc_fixed_results.json                # k=5    AUC=0.9671 (best_v2)
├── gsc_random_results.json               # k=5    AUC=0.9197 (best_v2)
├── triplet_gsc_fixed_k10.json            # k=10   AUC=0.9674 (best_v2)
├── triplet_gsc_random_k10.json           # k=10   AUC=0.9215 (best_v2)
├── kshot_ablation.json                   # k=1..20 ablation (best_v1)
├── denoiser_ablation.json                # EXT-1 (best_v1)
├── streaming_latency.json                # EXT-2 (model-agnostic)
├── gsc_fixed_k20_results.json            # k=20  AUC=0.9253 (best_v1)
├── gsc_fixed_k20_scaled_l2_results.json  # AUC=0.9118 (best_v1)
├── gsc_fixed_scaled_l2_results.json      # AUC=0.8549 (best_v1)
├── gsc_fixed_probability_results.json    # AUC=0.8041 (best_v1)
├── gsc_fixed_energy_results.json         # AUC=0.9114 (best_v1)
├── *.png                                 # DET curves, training curves
├── v1_margin0.5/
│   ├── gsc_fixed_results.json            # AUC=0.9098 (baseline để so sánh)
│   ├── gsc_random_results.json           # AUC=0.8751
│   ├── triplet_gsc_fixed_k10.json        # AUC=0.9229
│   └── triplet_gsc_random_k10.json       # AUC=0.8819
└── calibrated/
    └── gsc_fixed_results.json            # AUC=0.9515 (per-class threshold trên best_v1)
```

### Đã xóa

| Loại | Số file | Tổng dung lượng |
|---|---|---|
| Checkpoint intermediate (epoch_05/10/15/20/25/35.pt) trong cả 2 thư mục | 9 | ~45 MB |
| Checkpoint duplicate (`checkpoints/best.pt`, `kws_checkpoints_results/.../best.pt`, `latest.pt`) | 3 | ~15 MB |
| Result duplicate hash (gsc_fixed_energy_t05, openmax, openmax_nonorm) | 3 | ~232 KB |
| Result no-normalize đã bỏ (gsc_fixed_nonorm, gsc_fixed_energy_nonorm) | 2 | ~160 KB |
| Result trùng số (triplet_gsc_*_k5 trong `results/`) | 2 | ~158 KB |
| Thư mục `kws_checkpoints_results/` (sau khi move xong) | — | — |
| **Tổng** | **19 file + 1 dir** | **~60 MB** |

---

## File code đã được cập nhật default checkpoint

Tất cả 11 file dưới đây đã đổi `checkpoints/best.pt` (hoặc `checkpoints/triplet/best.pt`, hoặc `kws_checkpoints_results/.../best.pt`) → `checkpoints/triplet/best_v2_margin1.0_colab.pt`:

- `demo_web.py`
- `demo_quick.py`
- `calibrate_threshold.py`
- `test_paper_benchmark.py`
- `src/demo/app.py`
- `scripts/evaluate.py` (docstring)
- `scripts/inspect_checkpoints.py`
- `scripts/benchmark_denoiser.py`
- `scripts/benchmark_streaming.py`
- `scripts/compare_kshot.py`
- `scripts/test_long_audio_oracle.py`
- `README.md` (3 ví dụ dòng lệnh)

---

## Việc còn lại (optional, không bắt buộc)

| # | Việc | Lý do |
|---|------|-------|
| (a) | Re-run **kshot_ablation** + **k=20** + **scaled_l2/probability/energy ablation** với `best_v2` | Cập nhật để toàn bộ ablation đồng nhất trên cùng 1 encoder. |
| (b) | Re-run **per-class threshold calibration** trên `best_v2` | Hiện `calibrated/` đang dùng best_v1 (AUC 0.9515). Với best_v2 (AUC 0.9671 baseline) có thể đạy ≥ 0.97. |
| (c) | Re-run **denoiser_ablation** với `best_v2` | Để báo cáo EXT-1 đồng bộ. |
| (d) | Re-run **demo screenshots** với `best_v2` | Demo Gradio sẽ tốt hơn rõ rệt. |

→ **Không bắt buộc** ngay, có thể làm cuốn chiếu khi viết luận văn.

---

## Cấu trúc tóm tắt cho luận văn

> **Encoder mặc định:** `checkpoints/triplet/best_v2_margin1.0_colab.pt`
> (DSCNN-L, 35 epoch trên MSWC EN, triplet semi-hard mining, margin=1.0,
>  CosineAnnealingWarmRestarts, val_auc = 0.9395)

> **Baseline cũ:** `checkpoints/triplet/best_v1_margin0.5_local.pt`
> (đặt trong `results/v1_margin0.5/` để so sánh trước/sau retrain)
