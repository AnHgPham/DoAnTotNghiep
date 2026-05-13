# Phase 2 Hard-Pair Mining: Negative Result

> **Bottom line:** Hard-pair mining từ confusion matrix tính trên test set (GSC) đã làm encoder **overfit lên cặp được boost** và **suy giảm tổng thể** trên test data. **best_v2** vẫn là checkpoint tốt nhất cho production.

---

## 1. Setup

- **Checkpoint khởi điểm:** `best_v2_margin1.0_colab.pt` (epoch 30, val_auc trên MSWC val = 0.9395)
- **Hard pairs:** 21 cặp directional từ `results/hard_pairs.json`, sinh từ confusion matrix trên **GSC test set** (35 keyword × 100 query, k=5).
- **Sampler:** `EpisodicBatchSampler` với `hard_pair_prob=0.4` — 40% episodes ép cả 2 phần tử cặp khó vào batch.
- **Training:** Resume từ epoch 30 → 55 (25 epoch thêm), Colab A100, MSWC train 450 words × 500 sample.
- **Output:** `best_v3_margin1.0_phase2.pt` (epoch 54, val_auc trên MSWC val = **0.9771**)

## 2. Quan sát chính — `val_auc` MSWC tăng nhưng GSC giảm

| Metric | best_v2 | best_v3 | Δ |
|---|---|---|---|
| MSWC val_auc | 0.9395 | **0.9771** | **+3.76 pp** |
| MSWC val_acc | 0.529 | **0.910** | +38.1 pp |
| **GSC fixed AUC (k=5, 5 runs)** | **0.9708** | **0.9270** | **−4.4 pp** ❌ |
| GSC fixed FRR@5%FAR | 0.170 | 0.334 | +16.4 pp ❌ |
| GSC fixed Keyword ACC | 0.840 | 0.752 | −8.8 pp ❌ |
| GSC random AUC (k=5) | 0.9197 | 0.8870 | −3.27 pp ❌ |
| GSC random Keyword ACC | 0.902 | 0.837 | −6.5 pp ❌ |
| **GSC closed-set top-1 (35-way)** | **0.741** | **0.631** | **−11.0 pp** ❌ |

→ Cải thiện trên **MSWC val** không generalize lên **GSC test**. Đây là **biểu hiện điển hình của overfitting do data leakage qua confusion matrix**.

## 3. Hard pairs **được fix** nhưng tạo regression mới

Hard-pair mining đã đẩy lùi đúng các cặp được boost:

| Confusion pair | best_v2 (count) | best_v3 (count) | Reduction |
|---|---|---|---|
| `no → go` | 63 | **2** | **-97%** ✓✓ |
| `follow → forward` | 74 | 13 | -82% ✓ |
| `down → go` | 64 | 40 | -38% ✓ |
| `sheila → zero` | 89 | 59 | -34% ✓ |
| `three → tree` | 92 | 76 | -17% ✓ |
| `four → forward` | 87 | 72 | -17% ✓ |

Nhưng **xuất hiện confusions mới** mà best_v2 không có (hoặc rất ít):

| New confusion | best_v2 (count) | best_v3 (count) | NEW |
|---|---|---|---|
| `house → off` | ≤2 | **38** | NEW major |
| `right → one` | <5 | **30** | NEW |
| `follow → on` | ≤2 | **20** | NEW |
| `on → four` | <5 | **21** | NEW |
| `learn → no` | 8 | 20 | tăng |

## 4. Per-class accuracy collapse (trên 35 GSC keyword)

| Word | best_v2 | best_v3 | Δ |
|---|---|---|---|
| `right` | 0.96 | **0.60** | **−36 pp** ❌ |
| `house` | 0.95 | **0.51** | **−44 pp** ❌ |
| `one` | 0.85 | **0.56** | −29 pp ❌ |
| `dog` | 0.81 | **0.60** | −21 pp |
| `eight` | 0.97 | 0.91 | −6 pp |

→ Các keyword **vốn rất mạnh trong best_v2** (right 96%, house 95%) bị suy giảm nghiêm trọng. Encoder đã "đánh đổi" sự mạnh trên các cặp dễ để chỉnh các cặp khó.

## 5. Nguyên nhân: **task interference** trong episodic meta-learning

1. **Data leakage qua confusion matrix:** hard pairs được tính trên GSC test → bias sampling vào các cặp này = quasi-supervised signal từ test set lên train set. Encoder học specifically cho GSC distribution **của các cặp đó**, nhưng lại bị méo phần còn lại.

2. **Gradient interference:** trong triplet loss với episodic batch, ép cùng 2 class khó vào batch → triplet (anchor, positive, negative) có signal mạnh hơn cho cặp đó, nhưng **gradient mạnh hơn đè lên embedding space chung** → keyword khác bị xáo.

3. **Distribution shift train (MSWC) ↔ test (GSC):** MSWC và GSC có speaker, accent, phonetic profile khác nhau. Confusion pattern khác nhau giữa 2 dataset → boost theo GSC làm encoder lệch lên MSWC.

4. **MSWC val tăng vọt (0.91 acc):** vì MSWC có nhiều từ rare có cùng đặc tính như hard pairs → encoder học MSWC tốt nhưng phải đánh đổi GSC.

## 6. Bài học cho phương pháp luận

| Lesson |
|---|
| **Đừng dùng confusion matrix tính trên test set để bias training.** Đây là dạng test-set leakage tinh vi. |
| **Hard-pair mining nên dùng confusion từ một val set độc lập với cả train và test**, hoặc chỉ trên train. |
| **Validate trên cùng distribution với deployment** (GSC) thay vì MSWC val — `val_auc` Phase 2 không đáng tin nếu deployment là GSC. |
| **Cosine warm restart** trong Phase 2 không restart trong khoảng [30, 55] (chu kỳ thứ 3 đi 30→70) → LR cao đầu chu kỳ làm encoder thay đổi mạnh, ấp ủ overfitting. |

## 7. Kết luận & action

- **Production / báo cáo chính:** dùng **`best_v2_margin1.0_colab.pt`** (val_auc=0.9395, GSC AUC=0.97, KW-ACC=0.84).
- **`best_v3_margin1.0_phase2.pt`** giữ trong repo như **ablation study negative result** — minh chứng giới hạn của hard-pair mining khi không kiểm soát data leakage.
- **Phase 3 nếu muốn thử lại:**
  - Sinh confusion matrix từ một held-out subset của MSWC (không phải GSC).
  - Hoặc dùng **focal loss** thay hard-pair sampling — phân phối gradient theo confidence thay vì class identity.
  - Hoặc **augmentation focused** trên class yếu (speed perturb, noise mixing) thay vì sampling boost.

## 8. Files

| File | Mô tả |
|---|---|
| `checkpoints/triplet/best_v2_margin1.0_colab.pt` | Production checkpoint |
| `checkpoints/triplet/best_v3_margin1.0_phase2.pt` | Phase 2 result (overfit, không dùng default) |
| `results/gsc_fixed_results.json` | best_v2 trên gsc_fixed (AUC=0.97) |
| `results/v3_gsc_fixed/gsc_fixed_results.json` | best_v3 trên gsc_fixed (AUC=0.93) |
| `results/v3_gsc_random/gsc_random_results.json` | best_v3 trên gsc_random (AUC=0.89) |
| `results/confusion_matrix_best_v2_margin1.0_colab.json` | Closed-set 35-way: 0.741 |
| `results/confusion_matrix_best_v3_margin1.0_phase2.json` | Closed-set 35-way: 0.631 |
| `docs/phase2_negative_result.md` | Báo cáo này |
