# Runbook đạt kết quả tốt nhất - 3 luồng song song (2026-06-06)

Mục tiêu: chạy song song 3 luồng không trùng nhau để hoàn thiện bằng chứng thesis.

- Luồng A (server ict6, free): DSCNN-L + PCEN + SCAF+GE2E trên Top500.
- Luồng B (Colab A100 #2, ưu tiên): SCAF+GE2E ở quy mô ~2M (cap220), EdgeSpot và DSCNN.
- Luồng C (Colab A100 #1, thăm dò): KD teacher Wav2Vec2 cho EdgeSpot.
- Việc nhanh (bất kỳ session nào có GSC): eval cap620 @1%FAR.

QUAN TRỌNG - trước khi chạy Colab: phải push 2 file mới/sửa lên GitHub để bản clone trên Colab có chúng:
`scripts/train_teacher_head.py` (mới) và `scripts/precompute_teacher_embeddings.py` (đã sửa hỗ trợ FLAC + `--train-files`).

---

## Luồng A - Server ict6: DSCNN-L + PCEN + SCAF+GE2E trên Top500

Đăng nhập và chạy nền trong tmux (K80 chậm nhưng miễn phí):

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
cd /storage/<user>/an_kws/DoAnTotNghiep
conda activate kws_cu102
export CUDA_VISIBLE_DEVICES=4   # đổi sang GPU rảnh nếu cần (5/6/7 thường trống)

tmux new -s kws_top500_dscnn_scafge2e
```

Trong tmux:

```bash
cd /storage/<user>/an_kws/DoAnTotNghiep
conda activate kws_cu102
export CUDA_VISIBLE_DEVICES=4

LOG=/storage/<user>/an_kws/logs/top500_dscnn_scafge2e_e20_ep200.log
TAG=dscnn_pcen_scafge2e_top500_full_e20_ep200

python scripts/train.py --config configs/default.yaml \
  --data-dir data/mswc_top500_full \
  --model-family dscnn --feature-type mel_pcen --loss scaf_ge2e \
  --train-files train_files.json --val-files val_files.json \
  --epochs 20 --episodes 200 --n-classes 30 --n-samples 10 --max-per-word 0 \
  --num-workers 0 --run-tag $TAG \
  --select-by-gsc-dev --gsc-dev-every 2 --gsc-dev-runs 5 --gsc-dev-k-shot 10 \
  --save-every 1 --save-latest-every-epoch 2>&1 | tee $LOG

# Eval test100 ở cả 5% và 1% FAR
for FAR in 0.05 0.01; do
  python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/$TAG/best.pt \
    --model-family dscnn --feature-type mel_pcen \
    --protocol gsc_edgespot_exact --k-shot 10 --n-runs 100 --gsc-query-split test \
    --target-far $FAR --plot-det --output-dir results/$TAG/test100_far${FAR} 2>&1 | tee -a $LOG
done
```

Thoát tmux: `Ctrl+b` rồi `d`. Kiểm tra lại: `tmux attach -t kws_top500_dscnn_scafge2e`.

Ghi chú: dùng `--num-workers 0` để tránh lỗi fork hết RAM từng gặp trên K80.

---

## Luồng B - Colab A100 #2 (ƯU TIÊN): SCAF+GE2E quy mô ~2M (cap220)

Mở Colab A100 mới. Mount Drive + clone + cài deps:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!rm -rf DoAnTotNghiep
!git clone <REPO_GITHUB> DoAnTotNghiep
%cd /content/DoAnTotNghiep
```

B1. Chuẩn bị data cap220 FLAC (runner ở chế độ chỉ-chuẩn-bị, tắt train mặc định GE2E):

```bash
%%bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_heavy_flac_train.sh
MSWC_MAX_PER_WORD=220 RUN_DSCNN=0 RUN_EDGESPOT=0 bash colab/run_mswc_heavy_flac_train.sh
```

B2. Train SCAF+GE2E (cùng session, EdgeSpot ưu tiên):

```bash
%%bash
cd /content/DoAnTotNghiep
python scripts/train.py --config configs/default.yaml --data-dir data/mswc_en \
  --model-family edgespot_full --feature-type mel_pcen --edge-tau 4 --loss scaf_ge2e \
  --epochs 25 --episodes 1000 --n-classes 30 --n-samples 10 --max-per-word 0 \
  --train-files train_files_cap220_flac.json --val-files val_files_cap220_flac.json \
  --num-workers 8 --run-tag edgespot_t4_pcen_scafge2e_cap220_e25_ep1000 \
  --select-by-gsc-dev --gsc-dev-every 3 --gsc-dev-runs 5 --gsc-dev-k-shot 10 \
  --save-every 1 --save-latest-every-epoch
```

```bash
%%bash
cd /content/DoAnTotNghiep
# Tuỳ chọn nếu còn unit/thời gian: DSCNN + PCEN + SCAF+GE2E
python scripts/train.py --config configs/default.yaml --data-dir data/mswc_en \
  --model-family dscnn --feature-type mel_pcen --loss scaf_ge2e \
  --epochs 25 --episodes 1000 --n-classes 30 --n-samples 10 --max-per-word 0 \
  --train-files train_files_cap220_flac.json --val-files val_files_cap220_flac.json \
  --num-workers 8 --run-tag dscnn_pcen_scafge2e_cap220_e25_ep1000 \
  --select-by-gsc-dev --gsc-dev-every 3 --gsc-dev-runs 5 --gsc-dev-k-shot 10 \
  --save-every 1 --save-latest-every-epoch
```

B3. Eval 1% và 5% FAR + copy về Drive:

```bash
%%bash
cd /content/DoAnTotNghiep
for TAG in edgespot_t4_pcen_scafge2e_cap220_e25_ep1000 dscnn_pcen_scafge2e_cap220_e25_ep1000; do
  [ -f checkpoints/$TAG/best.pt ] || continue
  if echo $TAG | grep -q edgespot; then FAM=edgespot_full; EXTRA="--edge-tau 4"; else FAM=dscnn; EXTRA=""; fi
  for FAR in 0.05 0.01; do
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/$TAG/best.pt \
      --model-family $FAM --feature-type mel_pcen $EXTRA \
      --protocol gsc_edgespot_exact --k-shot 10 --n-runs 100 --gsc-query-split test \
      --target-far $FAR --plot-det --output-dir results/$TAG/test100_far${FAR}
  done
done
DST=/content/drive/MyDrive/DoAnTotNghiep_colab_runs/cap220_scafge2e_$(date +%Y%m%d)
mkdir -p $DST && cp -r results $DST/ && cp checkpoints/*scafge2e_cap220*/best.pt $DST/ 2>/dev/null || true
```

---

## Luồng C - Colab A100 #1 (THĂM DÒ): KD teacher Wav2Vec2 cho EdgeSpot

Mở Colab A100 mới riêng. Mount + clone + cài deps (giống Luồng B) và thêm transformers:

```bash
%cd /content/DoAnTotNghiep
!pip -q install transformers
```

C1. Chuẩn bị data cap220 FLAC (KD cần MSWC):

```bash
%%bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_heavy_flac_train.sh
MSWC_MAX_PER_WORD=220 RUN_DSCNN=0 RUN_EDGESPOT=0 bash colab/run_mswc_heavy_flac_train.sh
```

C2. Train teacher head (Wav2Vec2 + Sub-center ArcFace 64-D):

```bash
%%bash
cd /content/DoAnTotNghiep
python scripts/train_teacher_head.py --data-dir data/mswc_en \
  --train-files train_files_cap220_flac.json --max-per-word 50 \
  --epochs 30 --output outputs/teacher_head/teacher_head.pt
```

Kiểm tra log: dòng cuối phải có `best_train_top1` cao (head đã học, không ngẫu nhiên).

C3. Precompute teacher embeddings cho TOÀN BỘ file train của student (dùng cùng manifest):

```bash
%%bash
cd /content/DoAnTotNghiep
python scripts/precompute_teacher_embeddings.py --data-dir data/mswc_en \
  --train-files train_files_cap220_flac.json \
  --head-checkpoint outputs/teacher_head/teacher_head.pt \
  --batch-size 32 --output-dir outputs/teacher_w2v2_cap220
```

C4. Train student EdgeSpot với KD, eval 1%/5% FAR:

```bash
%%bash
cd /content/DoAnTotNghiep
TAG=edgespot_t4_pcen_kdscaf_cap220_e25_ep1000
python scripts/train.py --config configs/default.yaml --data-dir data/mswc_en \
  --model-family edgespot_full --feature-type mel_pcen --edge-tau 4 --loss kd_scaf \
  --teacher-embeddings-dir outputs/teacher_w2v2_cap220 \
  --epochs 25 --episodes 1000 --n-classes 30 --n-samples 10 --max-per-word 0 \
  --train-files train_files_cap220_flac.json --val-files val_files_cap220_flac.json \
  --num-workers 8 --run-tag $TAG \
  --select-by-gsc-dev --gsc-dev-every 3 --gsc-dev-runs 5 --gsc-dev-k-shot 10 \
  --save-every 1 --save-latest-every-epoch

for FAR in 0.05 0.01; do
  python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/$TAG/best.pt \
    --model-family edgespot_full --feature-type mel_pcen --edge-tau 4 \
    --protocol gsc_edgespot_exact --k-shot 10 --n-runs 100 --gsc-query-split test \
    --target-far $FAR --plot-det --output-dir results/$TAG/test100_far${FAR}
done
DST=/content/drive/MyDrive/DoAnTotNghiep_colab_runs/kd_cap220_$(date +%Y%m%d)
mkdir -p $DST && cp -r results $DST/ && cp -r outputs/teacher_head $DST/ && cp checkpoints/$TAG/best.pt $DST/ 2>/dev/null || true
```

Quyết định KD (theo docs/kd_research_plan_2026_06_06.md): KD chỉ giữ nếu trên GSC-test100 EdgeSpot tăng >= +1.0pp ACC@1%FAR, AUC không giảm, EER không xấu đi, F1 không giảm so với EdgeSpot GE2E cùng cap. Nếu không đạt thì bỏ KD, ghi là future work.

---

## Việc nhanh - Eval cap620 @1%FAR (chỉ cần GSC, ~30 phút)

Mở Colab bất kỳ, mount + clone + cài deps, tải GSC:

```bash
%%bash
cd /content/DoAnTotNghiep
python data/download_gsc.py --output-dir data/gsc_v2
RUNROOT=/content/drive/MyDrive/DoAnTotNghiep_colab_runs/colab_mswc_heavy_flac_target3000000_20260605_175317
DSCNN=$RUNROOT/checkpoints/dscnn_pcen_ge2e_cap620_flac_e20_ep1000_colab_mswc_heavy_flac_target3000000_20260605_175317/best.pt
EDGE=$RUNROOT/checkpoints/edgespot_full_t4_pcen_ge2e_cap620_flac_e20_ep1000_colab_mswc_heavy_flac_target3000000_20260605_175317/best.pt
python scripts/evaluate.py --config configs/default.yaml --checkpoint "$DSCNN" \
  --model-family dscnn --feature-type mel_pcen --protocol gsc_edgespot_exact \
  --k-shot 10 --n-runs 100 --gsc-query-split test --target-far 0.01 --plot-det \
  --output-dir results/cap620_dscnn_test100_far1
python scripts/evaluate.py --config configs/default.yaml --checkpoint "$EDGE" \
  --model-family edgespot_full --feature-type mel_pcen --edge-tau 4 --protocol gsc_edgespot_exact \
  --k-shot 10 --n-runs 100 --gsc-query-split test --target-far 0.01 --plot-det \
  --output-dir results/cap620_edgespot_test100_far1
cp -r results "$RUNROOT/results_far1_$(date +%Y%m%d)"
```

---

## Lưu ý vận hành

- Data Colab là local-only: Luồng B và C dùng 2 session riêng nên mỗi session tự chuẩn bị cap220 (~3h). Nếu muốn tiết kiệm unit, chạy B rồi C trong cùng 1 session (tuần tự, chậm hơn).
- Phải hoàn tất train trong cùng session (data mất khi session đóng). Đã bật `--save-latest-every-epoch` để resume nếu rớt.
- Sau khi có log, dán lại cho agent để cập nhật bảng kết quả thesis và đối chiếu SCAF+GE2E vs GE2E, KD vs non-KD.
