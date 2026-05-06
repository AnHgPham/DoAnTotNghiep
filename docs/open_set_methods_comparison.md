# Open-set rejection methods — empirical comparison

**Setup**: GSC Fixed protocol, k_shot=5, n_way=10 enrolled vs 20 unknown,
5 runs (mean ± std), DSCNN-L encoder (`checkpoints/triplet/best.pt`,
epoch 15, triplet loss = 0.108) with L2-normalized embeddings (276-dim).

Goal of this experiment: test whether alternative open-set rejection
methods can improve over the L2-distance baseline (`OpenNCM`) on the
existing encoder. The thesis baseline uses OpenNCM with per-prototype
threshold (`mean + 2·std` of support distances).

## Results

| Classifier | Variant | AUC | EER | FRR@5%FAR | Open-set ACC | Keyword ACC | F1 |
|------------|---------|-----|-----|-----------|--------------|-------------|----|
| **OpenNCM** | L2 distance, per-proto threshold | **0.9098 ± 0.0049** | **0.1674 ± 0.0103** | **0.3792 ± 0.0388** | **0.8116 ± 0.0155** | **0.7656 ± 0.0292** | **0.7684 ± 0.0133** |
| OpenMAX | per-class Weibull, argmax-score classify | 0.8533 ± 0.0182 | 0.2306 ± 0.0204 | 0.5284 ± 0.0394 | 0.7624 ± 0.0143 | 0.5872 ± 0.0587 | 0.6901 ± 0.0246 |
| OpenMAX | per-class Weibull, argmin-dist classify | 0.8241 ± 0.0200 | 0.2627 ± 0.0255 | 0.5508 ± 0.0411 | 0.7640 ± 0.0127 | 0.7656 ± 0.0292 | 0.6520 ± 0.0298 |
| OpenMAX | global Weibull + hybrid α=0.5 | 0.9098 ± 0.0049 | 0.1674 ± 0.0103 | 0.3792 ± 0.0388 | 0.8116 ± 0.0155 | 0.7656 ± 0.0292 | 0.7684 ± 0.0133 |
| Energy-OOD | T = 1.0 | 0.8188 ± 0.0059 | 0.2553 ± 0.0070 | 0.7212 ± 0.0248 | 0.6956 ± 0.0121 | 0.7656 ± 0.0292 | 0.6604 ± 0.0082 |
| Energy-OOD | T = 0.3 | 0.8811 ± 0.0103 | 0.1977 ± 0.0121 | 0.5636 ± 0.0615 | 0.7457 ± 0.0224 | 0.7656 ± 0.0292 | 0.7302 ± 0.0150 |
| Energy-OOD | T = 0.1 | 0.9124 ± 0.0064 | 0.1670 ± 0.0126 | 0.3812 ± 0.0541 | 0.8075 ± 0.0204 | 0.7656 ± 0.0292 | 0.7689 ± 0.0161 |
| Energy-OOD | T = 0.05 | 0.9114 ± 0.0052 | 0.1656 ± 0.0115 | 0.3736 ± 0.0423 | 0.8117 ± 0.0176 | 0.7656 ± 0.0292 | 0.7706 ± 0.0147 |

**Best alternative**: Energy-OOD T=0.05 — Δ over OpenNCM is +0.16% AUC,
−0.18% EER, +0.22% F1. All deltas lie **within one standard deviation
of the baseline**, so the improvement is not statistically significant
at 5 runs.

## Why nothing beats OpenNCM here

Three structural reasons explain the convergence:

1. **L2-normalized embedding space is bounded** (distances ∈ [0, 2]).
   Methods that rely on heavy-tailed distance distributions (Weibull
   in OpenMAX) lose discriminative power because there is no real
   "tail" — the worst case is bounded.

2. **Few-shot regime (k=5) starves Weibull tail estimation.** The
   `shape` parameter for the same class fluctuates between 2.4 and 6.0
   across runs because only 5 support distances are available per
   class. Per-class fits add noise; pooling into a global Weibull
   recovers OpenNCM exactly because the result becomes a monotonic
   transform of `−min(d)`.

3. **Energy-OOD reduces to OpenNCM as T → 0.** Empirically, lowering
   T from 1.0 to 0.05 monotonically improves AUC from 0.819 to 0.911.
   The limit is precisely `−min(d)`, i.e. OpenNCM. The hope was that
   moderate T would aggregate information across all prototypes and
   help discriminate "uniform unknowns" from "unimodal knowns", but
   in our setting unknown queries also collapse to one acoustically
   similar class (e.g. "five" near "nine"), so the multi-prototype
   signal does not separate them better than the single nearest one.

## Implication for the thesis

The bottleneck is the **encoder**, not the open-set scorer. With this
checkpoint and L2-normalized embeddings, all four families of open-set
methods we tried (NCM, Weibull-revised, energy-aggregated) converge to
~0.91 AUC. To push past this ceiling, future work needs to change one
of:

- **Encoder architecture** — e.g. BC-ResNet (98.0% GSC SOTA at 9.7k
  params) or KWT-3 (98.7% with pretrained weights). New encoder gives
  embeddings that *can* be separated more aggressively, at which point
  any of the open-set methods above would benefit.
- **Embedding norm** — train without L2 normalization so the
  unbounded magnitude itself carries OOD signal, or use ArcFace
  (`src/models/arcface.py`) which learns calibrated cosine margins.
- **Loss objective** — add an explicit OOD term (e.g. Outlier
  Exposure, OE) during training so the encoder produces low-confidence
  embeddings on out-of-keyword speech.

## Negative-result value

Rusci et al. (2024) discuss three open-set methods (OpenNCM, OpenMAX,
DProto) but only experiment with OpenNCM. This chapter empirically
fills that gap for OpenMAX and adds Energy-OOD (Liu et al. 2020) as a
third comparator. The finding that all three methods plateau on a
shared ceiling is a meaningful contribution — it isolates the
remaining improvement budget to the encoder side.

## Reproducibility

```bash
# Baseline
python scripts/evaluate.py --classifier openncm --protocol gsc_fixed --n-runs 5

# OpenMAX variants
python scripts/evaluate.py --classifier openmax --protocol gsc_fixed --n-runs 5
python scripts/evaluate.py --classifier openmax --openmax-mode global --openmax-hybrid 0.5 \
    --protocol gsc_fixed --n-runs 5

# Energy-OOD sweep
for T in 1.0 0.3 0.1 0.05; do
  python scripts/evaluate.py --classifier energy --energy-temperature $T \
      --protocol gsc_fixed --n-runs 5
done
```

Result JSONs land in `results/gsc_fixed_*.json`. DET PNGs are produced
with `--plot-det --det-output PATH`.

## Files added

- `src/classifiers/openmax.py`, `tests/test_openmax.py` — 12 tests pass.
- `src/classifiers/energy_ood.py`, `tests/test_energy_ood.py` — 12 tests pass.
- `src/evaluation/protocols.py` — added `"openmax"` and `"energy"` scoring branches.
- `scripts/evaluate.py` — added `--classifier {openncm,openmax,energy}`,
  `--openmax-mode`, `--openmax-hybrid`, `--tail-size`, `--energy-temperature`.
- OpenNCM tests still pass (10/10) — no regression in the existing path.

## Side experiment: Silero VAD vs energy-based segmentation (demo)

Hypothesis from the original improvement list: replacing the demo's
energy-threshold segmentation with Silero VAD would give +5–10%
accuracy. Tested on 4 long demo files (`data/test/gsc_demo_*.wav`)
with k=5 enrollment, L2 scoring, threshold=0.85:

| Audio | Energy + L2 | Silero VAD + L2 | Δ |
|-------|-------------|------------------|---|
| diverse | 65.0% | 55.0% | −10.0% |
| easy | 65.0% | 65.0% | 0.0% |
| numbers | 75.0% | 80.0% | +5.0% |
| names | 45.0% | 45.0% | 0.0% |
| **average** | **62.5%** | **61.25%** | **−1.25%** |

Silero VAD over-segments concatenated GSC clips on `diverse`. The demo
default (`Energy`) is therefore **kept** in `demo_web.py:832`. Silero
VAD remains available as a user-selectable option for streaming /
real-microphone use cases where it is known to behave better.
