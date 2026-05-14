# Tier-1 Research Upgrade Notes

## Objective

This upgrade turns the project from a DSCNN + Triplet baseline into a
paper-grade few-shot open-set KWS research pipeline.

The implementation is intentionally split into reproducible components:

- canonical EdgeSpot-style GSC protocol with true `_silence_`;
- EdgeSpotFull and BCResNetFS mel/PCEN encoders;
- GE2E and hybrid SCAF/GE2E/KD objectives;
- offline Wav2Vec2 teacher embedding precompute;
- prototype and threshold calibration utilities;
- report/table scripts for final experiments.

## What Is Implemented

- `gsc_edgespot_exact`: 10 GSC command targets plus true `_silence_`, with
  silence generated from `_background_noise_` crops.
- `GSCFewShotProvider(query_split="dev"|"test")`: dev uses official train files
  as query pool; test uses `testing_list.txt`.
- `EdgeSpotFull(tau=1..4)`: 40x101 mel, PCEN, fused temporal blocks,
  BC-ResNet-style residual blocks, temporal positional Conv1D, single-head SDPA,
  and 64-D embedding.
- `BCResNetFS`: no-attention architecture baseline.
- `GE2ELoss`: support/query split inside each episode, matching few-shot
  enrollment at deployment.
- KD scaffold: precompute Wav2Vec2 teacher embeddings once and train student
  models from shards with `kd_scaf`, `kd_ge2e`, or `kd_scaf_ge2e`.

## What Is Not Claimed Yet

- The code does not claim EdgeSpot paper numbers until full MSWC training and
  100-run GSC test evaluation are completed.
- The default Wav2Vec2 teacher projection is useful for smoke tests only unless
  a trained projection head checkpoint is provided.
- Streaming remains a downstream phase after static few-shot open-set metrics
  become stable.

## Canonical Commands

```bash
python scripts/model_report.py --family edgespot_full --tau 4

python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e

python scripts/evaluate_edgespot_protocol.py \
  --checkpoint checkpoints/edgespot_full_t4_scaf_ge2e/best.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir results/edgespot_exact/edgespot_full_t4_scaf_ge2e
```

## Acceptance Targets

- Reproduction acceptable: EdgeSpotFull tau=4 reaches at least 78% ACC@1% FAR.
- Reproduction successful: 80-82% ACC@1% FAR.
- Proposed method successful: GE2E/KD/calibration improves reproduction by at
  least 2 percentage points or materially lowers FRR@5% FAR.
