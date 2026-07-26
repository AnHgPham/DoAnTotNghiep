# Production Demo Handoff - 2026-07-11

## Objective

Raise the local KWS research demo to a reliable production-demo standard and remove perceptible DSCNN inference delay without changing the detection decision protocol.

## Flagship Demo Profiles

- Best accuracy / default:
  - Profile: `dscnn_pcen_ge2e`
  - Architecture: DSCNN-L + PCEN + GE2E
  - Checkpoint: `checkpoints/dscnn_pcen_ge2e_accdev_ep300_composite_colab_mswc.pt`
  - GSC test100 ACC@1%FAR: `86.36%`
  - Encoder parameters: `412,900`
- Best compact:
  - Profile: `edgespot_t4_pcen_ge2e`
  - Architecture: EdgeSpotFull T4 + PCEN + GE2E
  - Checkpoint: `checkpoints/edgespot_t4_pcen_ge2e_ep300_composite_colab_mswc_c.pt`
  - GSC test100 ACC@1%FAR: `82.87%`
  - Encoder parameters: `130,598`
- Both originate from 60-epoch runs with 300 training episodes per epoch and composite GSC-dev checkpoint selection.
- Local URL: `http://127.0.0.1:8000/`
- Health endpoint: `/api/health`

## Implemented

- Cached MFCC/Mel transform tensors and added vectorized batch feature extraction.
- Added batched/chunked embedding with a bounded default batch size.
- Batched enrollment views across keywords and rebuilt only changed keywords.
- Moved enrollment, decode, inference, and model initialization work off the FastAPI event loop.
- Serialized enrollment/model mutations with `STATE_MUTATION_LOCK`.
- Batched candidate windows across every long-audio speech segment while preserving candidate windows, thresholds, margins, votes, and cooldown behavior.
- Reused single-detection feature maps and robust-event distances instead of running duplicate inference.
- Added upload/duration limits, restricted CORS, startup warm-up, `/api/health`, and response `timing_ms`.
- Made the production DSCNN profile the default when its checkpoint is available.
- Corrected benchmark checkpoint/frontend/model-family handling.
- Replaced the 4 MB Material Symbols font with Lucide icons.
- Reduced the default model list to the two verified composite flagship profiles; older curated and auto-discovered profiles remain collapsible.
- Added latency and processing-speed metrics to the UI.
- Added a favicon and compatible `/favicon.*` and `/ui/favicon.svg` routes.
- Kept Python 3.9 compatibility for the ict6 CUDA 10.2 environment.

## Measured Local CPU Performance

- Single request over 17 unseen GSC samples:
  - median inference: `29.73 ms`
  - median server total: `33.56 ms`
  - median local HTTP wall time: `64.76 ms`
- Enrollment, 17 keywords x 5 shots: about `5.0 s` without synthetic polling load.
- Event-loop responsiveness during enrollment stress:
  - 288 health responses
  - median `5.51 ms`
  - zero errors
- Long audio, `22.93 s`:
  - before file-level batching: `4.50 s`
  - after file-level batching: `2.69 s`
  - `8.51x` real-time
  - accepted sequence remained unchanged in the before/after comparison.
- Existing smoke benchmark artifact: `reports/production_dscnn_benchmark_smoke.json`.
- Live two-profile switch check on local CPU:
  - EdgeSpot T4 composite switch: `148.5 ms`; median single request: `21.94 ms`.
  - DSCNN composite switch with enrollment rebuild: `561.9 ms`; median single request: `26.97 ms`.

## Verification

- Python: `165 passed, 1 torchaudio future warning`.
- UI: `npm run typecheck` passed.
- UI: `npm run build` passed with Vite `7.3.6`.
- Production dependency audit: `0 vulnerabilities`.
- Scoped `git diff --check`: passed.
- Chromium visual/API smoke:
  - exactly two flagship model cards are visible by default
  - older Top500, Microset, and legacy cards are hidden until `Show all`
  - desktop and mobile have no horizontal page overflow
  - desktop and mobile console errors: none
  - desktop and mobile HTTP errors: none
  - single latency and long-audio processing speed rendered successfully.
- Screenshots:
  - `.codex_tmp/screens/desktop-single-production.png`
  - `.codex_tmp/screens/desktop-long-production.png`
  - `.codex_tmp/screens/mobile-single-production.png`

## Claim Boundaries

- The 12/17 unseen-file smoke outcome is a demo regression check, not an accuracy result.
- UI sampled evaluation does not replace `gsc_edgespot_exact test100`.
- Candidate windows and decision logic were preserved intentionally; no accuracy claim is based on the performance refactor alone.
- This is an 8-8.5/10 local research production demo, not internet-facing production. Authentication, multi-user load testing, observability, and deployment hardening remain outside the current scope.

## Main Files Changed

- `src/demo/api_server.py`
- `src/features/mfcc.py`
- `src/features/mel.py`
- `src/streaming/enrollment.py`
- `src/streaming/robust_engine.py`
- `scripts/benchmark_robust_streaming.py`
- `src/demo/ui/src/App.tsx`
- `src/demo/ui/src/types.ts`
- `src/demo/ui/src/i18n.ts`
- `src/demo/ui/public/favicon.svg`
- focused tests under `tests/`

## Recommended Next Step

Keep this implementation frozen for the defense demo. If additional engineering time is available, add a repeatable concurrent load test and package the server as a single launch command/service. Do not tune thresholds against the small smoke samples; use the formal dev/calibration protocol and report test100 separately.
