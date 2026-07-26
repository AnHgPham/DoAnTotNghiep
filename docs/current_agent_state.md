# Current Agent State

Last updated: 2026-07-13 ICT

## 2026-07-13 Update - April-July Timeline And ICT6 Audit

Created the evidence-backed Vietnamese audit:

- `docs/reports/project_timeline_training_audit_2026_04_to_07_vi.md`

Critical reconciliation:

- ICTLab/ict6 was a substantial training environment, not only a dataset host.
- The server fixed40 matrix
  `full_mswc_16_pipeline_manifest20_e40_ep150_fixed` did not complete 16/16.
  The copied server TSV shows 1 completed row and 15 `failed_rc_1` rows.
- The completed fixed 16-pipeline matrix used by the thesis is the Colab A100
  cap620 run, followed by the three-run cap620 development stage.
- Raw ict6 audit files copied on 2026-07-13 are under
  `reports/server_metrics_raw/ict6_audit_20260713/`.
- The original pasted Colab fixed16 log was preserved at
  `reports/colab_cap620_fixed_raw/colab_mswc_cap620_fixed16_20260611_154517.txt`.
- The project timeline now separates completed training, eval-only work,
  failed/resumed jobs, log-only historical results and locked artifacts.

## 2026-06-14 Update - Reference-Style Vietnamese Thesis Draft

User requested restructuring the thesis to follow the referenced master thesis
format more closely.

Generated new reference-style thesis builder:

- `scripts/make_reference_style_thesis_vi_2026_06_14.py`

Generated outputs:

- Markdown:
  - `docs/thesis/Do_An_KWS_thesis_reference_style_vi_2026_06_14.md`
- Word:
  - `docs/thesis/Do_An_KWS_thesis_reference_style_vi_2026_06_14.docx`

Structure now follows the reference-style arc:

1. Introduction.
2. Background.
3. Methodology and experimental design.
4. Results and discussion.
5. Conclusions and future work.

Key changes from the older 2026-06-13 thesis:

- Added formal formula sections for ProtoNet, episodic training, direct L2
  decision, MFCC, PCEN, Triplet, GE2E, SCAF, FAR/FRR/ACC@FAR.
- Added explicit mapping between the reference master thesis and this project.
- Clarified that MSWC is the main training source in current evidence, while
  MSWC randomized evaluation exists in code but is not the main final evidence.
- Kept `gsc_edgespot_exact test100` as the main benchmark evidence.
- Moved demo UI to a bounded discussion item rather than making it a main
  scientific chapter.

Verification:

- `python scripts\make_reference_style_thesis_vi_2026_06_14.py` completed.
- `python -m py_compile scripts\make_reference_style_thesis_vi_2026_06_14.py`
  passed.
- DOCX inspected with `python-docx`:
  - `174` paragraphs.
  - `15` tables.
  - Vietnamese Unicode text readable.
- Markdown inspected:
  - `50` headings.
  - `30` math block delimiters.
  - main headings include Chương 1-5 and appendices.

Read this after `AGENTS.md` and `docs/agent_orchestration_init.md`.

## 2026-06-13 Update - Cap620 Development Run Completed

User completed Colab development run:

- Run ID:
  - `colab_mswc_cap620_development_20260612_050614`
- Source log:
  - attachment `pasted-text.txt`
- Report written:
  - `docs/reports/cap620_development_20260612_summary_vi.md`

Run configuration:

- Runtime: Colab A100 40GB.
- Data: MSWC English cap620 FLAC.
- Train files: `2,989,780`.
- Validation files: `52,399`.
- Train words: `37,387`.
- Validation words: `763`.
- Toggles:
  - `RUN_ACCURACY=1`
  - `RUN_COMPACT=1`
  - `RUN_KD=0`
  - `RUN_SCAF_ABLATION=0`
- Checkpoint selection:
  - GSC-dev composite metric = mean(`ACC@1%FAR`, `AUC`, `F1`).
- Final eval:
  - `dev30_far1`
  - `test100_far1`
  - `test100_far5`

Completed stages:

- `DSCNN-L + PCEN + GE2E`, 60 epochs, 300 episodes/epoch,
  `40 classes x 10 samples`, hard-pair seeding.
- `EdgeSpotFull T4 + PCEN + Triplet hard`, 60 epochs, 300 episodes/epoch.
- `EdgeSpotFull T4 + PCEN + GE2E`, 60 epochs, 300 episodes/epoch.

New best test100 results:

- Best overall:
  - `DSCNN-L + PCEN + GE2E, ep300 composite`
  - `ACC@1%FAR=86.36 +/- 1.29`
  - `AUC=95.21 +/- 0.45`
  - `EER=11.32 +/- 0.78`
  - `F1=82.73 +/- 1.11`
  - `ACC@5%FAR=89.93 +/- 0.65`
- Best compact:
  - `EdgeSpotFull T4 + PCEN + GE2E, ep300 composite`
  - `ACC@1%FAR=82.87 +/- 1.22`
  - `AUC=92.41 +/- 0.44`
  - `EER=14.82 +/- 0.70`
  - `F1=77.85 +/- 0.97`
  - `ACC@5%FAR=86.76 +/- 0.59`
- Failed/collapsed compact branch:
  - `EdgeSpotFull T4 + PCEN + Triplet hard`
  - `ACC@1%FAR=69.10 +/- 0.15`
  - `AUC=53.40 +/- 0.48`
  - `EER=47.84 +/- 0.62`
  - `F1=39.99 +/- 0.60`
  - Do not use this as evidence that Triplet is bad in general; the fixed
    baseline Triplet was much better. Likely issue is overly aggressive
    hard mining plus hard-pair episode seeding.

Updated claim boundary:

- `DSCNN-L + PCEN + GE2E` now clearly improves over the fixed cap620 result
  and is the main accuracy result.
- `EdgeSpotFull T4 + PCEN + GE2E` now has mean `ACC@1%FAR=82.87`, which is
  above the EdgeSpot-4 paper reference mean `82.0`, but the margin is modest
  (`+0.87`) and within the reported run-to-run std. Phrase as:
  "competitive and slightly above the reported EdgeSpot-4 mean under our
  GSC test100 protocol", not as a definitive paper reproduction.
- KD was not run (`RUN_KD=0`), so do not claim the EdgeSpot paper's
  teacher-distillation recipe has been reproduced.

## 2026-06-12 Update - Cap620 Development Experiment Runner Configured

User requested follow-up experiment configuration after the fixed cap620
16-pipeline result:

- Accuracy target:
  - optimize `DSCNN-L + PCEN + GE2E`;
  - increase episode budget;
  - use hard episode mining;
  - select checkpoints by composite `ACC@1%FAR + AUC + F1`.
- Compact target:
  - optimize `EdgeSpotFull T4 + PCEN + Triplet`;
  - optimize `EdgeSpotFull T4 + PCEN + GE2E`;
  - avoid SCAF for full cap620 because cap620 SCAF collapsed.
- EdgeSpot-4 comparison target:
  - add KD / teacher-guided objective as an opt-in branch, because the
    EdgeSpot paper uses distillation from a self-supervised teacher.
- SCAF target:
  - run subset ablation only: loss weight, margin, scale, warmup, subset size.

Implemented:

- `colab/run_mswc_cap620_development_experiments.sh`
- `docs/colab/cap620_development_experiments_runbook_vi.md`

Training script support added:

- `scripts/train.py`
  - `--gsc-dev-select-metric {acc1far,composite}`
  - composite metric = mean of GSC-dev `ACC@1%FAR`, `AUC`, `F1`
  - `--train-words-file`, `--val-words-file`
  - `--resume-encoder-only`
  - `--arcface-scale`, `--arcface-margin`, `--arcface-sub-centers`
- `scripts/precompute_teacher_embeddings.py`
  - `--train-words-file`
  - manifest-based teacher precompute can now filter to KD subset
  - `--max-per-word` works with manifest precompute
- `scripts/train_teacher_head.py`
  - `--train-words-file`
  - teacher head and KD student can now use the same subset word file

Default Colab development runner behavior:

- `RUN_ACCURACY=1`
  - runs `DSCNN-L + PCEN + GE2E`
  - defaults: `60` epochs, `300` episodes/epoch,
    `40 classes x 10 samples`
  - hard-pair seeding from `results/hard_pairs.json`
  - checkpoint selection by composite GSC-dev metric
- `RUN_COMPACT=1`
  - runs `EdgeSpotFull T4 + PCEN + Triplet`
  - runs `EdgeSpotFull T4 + PCEN + GE2E`
  - defaults: `60` epochs, `300` episodes/epoch,
    `40 classes x 10 samples`
- `RUN_KD=0`
  - opt-in only; trains Wav2Vec2 teacher head, precomputes teacher embeddings
    on a subset, then trains `kd_ge2e`
- `RUN_SCAF_ABLATION=0`
  - opt-in only; subset GE2E warmup then `scaf_ge2e` finetune with lower
    SCAF weight/scale/margin

Verification:

- `python -m py_compile scripts\train.py scripts\precompute_teacher_embeddings.py scripts\train_teacher_head.py`
  passed.
- `python scripts\train.py --help` shows the new train options.
- `python scripts\precompute_teacher_embeddings.py --help` shows
  `--train-words-file`, `--max-per-word`, `--seed`.
- `python scripts\train_teacher_head.py --help` shows
  `--train-words-file`.
- Git Bash `bash -n colab/run_mswc_cap620_development_experiments.sh` passed.

## 2026-06-12 Update - Final Vietnamese Thesis Draft Generated

User requested a standard thesis, preferably Word, based on the completed
cap620 fixed 16-pipeline experiment and the existing thesis material.

Generated reproducible thesis builder:

- `scripts/make_final_thesis_vi_2026_06_12.py`

Generated outputs:

- Markdown:
  - `docs/thesis/Do_An_KWS_final_vi_2026_06_12.md`
- Word:
  - `docs/thesis/Do_An_KWS_final_vi_2026_06_12.docx`
- Audit:
  - `docs/thesis/Do_An_KWS_final_vi_2026_06_12_audit.txt`
- Figures:
  - `docs/thesis/assets_final_2026_06_12/cap620_top8_acc1far.png`
  - `docs/thesis/assets_final_2026_06_12/cap620_acc1far_heatmap.png`
  - `docs/thesis/assets_final_2026_06_12/edgespot4_comparison_acc1far.png`

Verification:

- Generator completed successfully with `python-docx` and `pandas`.
- DOCX inspected with `python-docx`:
  - `146` paragraphs.
  - `14` tables.
  - `3` embedded figures.
  - `16` headings.
- Audit file confirms:
  - `48` CSV rows.
  - `16` unique pipelines.
  - evals: `dev30_far1`, `test100_far1`, `test100_far5`.
  - all train/dev/test statuses are `ok`.
  - best `test100_far1`: `DSCNN-L + PCEN + GE2E`,
    `ACC=82.34`, `AUC=92.42`, `EER=14.89`, `F1=77.75`.

Thesis content stance:

- Cap620 fixed 16-pipeline is the main evidence.
- Microset and Top500 are included only as context with claim boundaries.
- EdgeSpot-4 comparison is phrased conservatively:
  - project best overall is competitive/slightly above paper using larger
    DSCNN-L;
  - compact EdgeSpotFull T4 cap620 does not beat EdgeSpot-4 paper.
- SCAF/SCAF+GE2E collapse is explicitly explained; reject-all ACC around
  `69.44` is not treated as good.

## 2026-06-12 Update - Colab Cap620 Fixed 16-Pipeline Completed And Thesis Chapter Drafted

User completed the Colab fixed 16-pipeline cap620 FLAC run:

- Run ID:
  - `colab_mswc_cap620_flac_16pipe_e40_ep150_20260611_154517`
- Drive artifact:
  - `/content/drive/MyDrive/DoAnTotNghiep_colab_runs/colab_mswc_cap620_flac_16pipe_e40_ep150_20260611_154517`
- Local extracted metrics:
  - `results/cap620_16_pipeline_metrics_long.csv`
  - `results/cap620_16_pipeline_test100_summary.md`
- Completion evidence:
  - `16` unique pipelines.
  - `48` metric rows: `dev30_far1`, `test100_far1`, `test100_far5` for each pipeline.
  - all statuses are `ok`: `train`, `dev30_far1`, `test100_far1`, `test100_far5`.

Cap620 fixed protocol data/training constants:

- Data profile: MSWC English cap620 FLAC.
- Train files: `2,989,780`.
- Validation files: `52,399`.
- Train words: `37,387`.
- Validation words: `763`.
- Epochs: `40`.
- Episodes/epoch: `150`.
- Episode batch: `30 classes x 10 samples`.
- Checkpoint selection: GSC-dev `ACC@1%FAR`, every 5 epochs, 3 runs, k=10.
- Final evaluation: GSC `dev30_far1`, `test100_far1`, `test100_far5`.

Main cap620 test100 findings:

- Best overall:
  - `DSCNN-L + PCEN + GE2E`
  - test100 FAR1:
    - `ACC@1%FAR=82.34 +/- 1.19`
    - `AUC=92.42 +/- 0.54`
    - `EER=14.89 +/- 0.84`
    - `F1=77.75 +/- 1.15`
  - test100 FAR5:
    - `ACC@5%FAR=86.57 +/- 0.75`
    - `FRR@5%FAR=29.18 +/- 2.60`
- Best EdgeSpotFull T4 by ACC@1%FAR:
  - `EdgeSpotFull T4 + PCEN + GE2E`
  - `ACC@1%FAR=79.98 +/- 0.98`
  - `AUC=87.23 +/- 0.75`
  - `EER=20.23 +/- 0.96`
  - `F1=70.68 +/- 1.23`
- Best EdgeSpotFull T4 by AUC/EER/F1:
  - `EdgeSpotFull T4 + PCEN + Triplet`
  - `ACC@1%FAR=79.58 +/- 1.35`
  - `AUC=89.85 +/- 0.63`
  - `EER=18.22 +/- 0.78`
  - `F1=73.29 +/- 1.02`
- SCAF/SCAF+GE2E mostly collapses on cap620:
  - several rows show AUC about `50`, EER about `50`, FRR@FAR `100`, F1 `0`.
  - Do not interpret `open-set ACC ~=69.44` on those rows as good; it is reject-all behavior.

Paper comparison note:

- EdgeSpot paper arXiv `2601.16316` reports EdgeSpot-4:
  - `10-shot ACC@1%FAR=82.0`
  - `128k` params
  - `29.4M` MACs
- Cap620 fixed claim boundary:
  - Overall best `DSCNN-L + PCEN + GE2E` is roughly competitive and slightly above `82.0`, but it is a larger DSCNN model and the margin is within std.
  - Project `EdgeSpotFull T4` cap620 fixed best is `79.98`, so it does **not** beat EdgeSpot-4 paper under this protocol.
  - Top500 epoch13 EdgeSpot artifact has `ACC@1%FAR=85.62`, but it is a separate profile and must not be mixed into the cap620 fixed ranking.

New Vietnamese thesis/scientific chapter drafted:

- `docs/thesis/cap620_16_pipeline_scientific_chapter_vi_2026_06_12.md`

This chapter covers:

- fixed 16-pipeline protocol,
- cap620 FLAC data config,
- training config,
- GSC dev/test100 evaluation config,
- FAR1/FAR5 definitions and reporting logic,
- full test100 tables,
- why PCEN/GE2E/Triplet help,
- why SCAF collapses,
- EdgeSpot-4 paper comparison,
- claim boundaries and future work.

## 2026-06-11 10:25 ICT Update - Fixed 16-Pipeline 40-Epoch Protocol Started

User requested a standardized 16-pipeline test with one fixed epoch setting so
future comparisons are not mixed with ad-hoc runs.

Implemented fixed protocol files:

- `server/run_full_mswc_16_pipeline_manifest20_e40_fixed.sh`
- `server/launch_full_mswc_16_pipeline_manifest20_e40_fixed.sh`
- `docs/full_mswc_16_pipeline_e40_fixed_protocol_vi.md`

Protocol constants are hard-coded in the runner:

- Run ID: `full_mswc_16_pipeline_manifest20_e40_ep150_fixed`
- Dataset: `data/mswc_en`
- Manifests:
  - `data/mswc_en/splits/train_files.json`: `527069` files
  - `data/mswc_en/splits/val_files.json`: `10637` files
- Epochs: `40`
- Episodes/epoch: `150`
- Episode batch: `30 classes x 10 samples`
- Max per word: `20`
- Checkpoint selection: GSC-dev `ACC@1%FAR`, every `5` epochs, `3` runs, `k=10`
- Final eval per pipeline:
  - `dev30_far1`
  - `test100_far1`
  - `test100_far5`

The full matrix is exactly:

- DSCNN-L x MFCC/PCEN x Triplet/SCAF/GE2E/SCAF+GE2E = 8
- EdgeSpotFull T4 x MFCC/PCEN x Triplet/SCAF/GE2E/SCAF+GE2E = 8

Server deployment:

- Files copied to ict6 repo and `bash -n` passed for both shell scripts.
- Launched tmux session:
  - `kws_mswc16_e40_fixed`
- Wait log:
  - `/storage/<user>/an_kws/logs/full_mswc_16_pipeline_manifest20_e40_ep150_fixed_wait_gpu.log`
- Run log:
  - `/storage/<user>/an_kws/logs/full_mswc_16_pipeline_manifest20_e40_ep150_fixed.log`
- Summary TSV:
  - `/storage/<user>/an_kws/logs/full_mswc_16_pipeline_manifest20_e40_ep150_fixed_runs.tsv`
- Evidence at launch:
  - waiter selected GPU `1`
  - first process started:
    `python scripts/train.py ... --model-family dscnn --feature-type mfcc --loss triplet --epochs 40 --episodes 150 ...`

Reporting rule:

- Do not mix this fixed 40-epoch matrix with the old 5-epoch phase-1 table or
  the 20-epoch shortlist table. Report those as screening/shortlist only.

## 2026-06-11 Update - Colab Cap620 16-Pipeline Runner Prepared

User confirmed Colab units are sufficient and wants the same fixed 16-pipeline
protocol on the larger cap620 FLAC data profile.

Implemented:

- `colab/run_mswc_cap620_16_pipeline_e40_fixed.sh`
- `colab/package_code_for_colab_posix.ps1`

Packaged upload-ready POSIX zip:

- `D:\Downloads\DoAnTotNghiep\DoAnTotNghiep_code_colab_cap620_16pipe_POSIX.zip`

Zip entry verification showed POSIX `/` paths, including:

- `DoAnTotNghiep/colab/run_mswc_cap620_16_pipeline_e40_fixed.sh`
- `DoAnTotNghiep/data/convert_opus_to_flac.py`
- `DoAnTotNghiep/scripts/evaluate.py`
- `DoAnTotNghiep/scripts/train.py`

The Colab runner hard-codes:

- Data profile: `MSWC cap620 FLAC`
- Epochs: `40`
- Episodes/epoch: `150`
- Episode batch: `30 classes x 10 samples`
- GSC checkpoint selection: dev `ACC@1%FAR`, every `5` epochs, `3` runs
- Final eval: `dev30_far1`, `test100_far1`, `test100_far5`
- Matrix: DSCNN-L and EdgeSpotFull T4 x MFCC/PCEN x
  Triplet/SCAF/GE2E/SCAF+GE2E = 16 pipelines

Important comparison rule:

- Server fixed run is `manifest20/cap20`, about `527,069` train files.
- Colab fixed run is `cap620 FLAC`, about `2,989,780` train files and
  `52,399` validation files if it matches the previous cap620 run.
- Compare pipelines within the same data profile. Do not directly mix cap20
  and cap620 numbers in one ranking table without labeling the data profile.

## 2026-06-06 Update - KD Research Plan Saved

Saved the KD research/re-ranking plan for later use:

- `docs/kd_research_plan_2026_06_06.md`

Key point: KD should only affect thesis ranking if the teacher is valid, not a random projection head, and if KD improves `GSC-test100` against the non-KD baselines.

## 2026-06-06 00:35 ICT Update - Server FAR1 Eval And Colab Package

Server ict6:

- Top500Full EdgeSpotFull T4 + PCEN + SCAF+GE2E resume has already completed:
  - summary:
    `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_edgespot_resume_runs.tsv`
  - status: `train=ok`, `dev=ok`, `test=ok`
  - completed: `2026-06-05 04:58:56`
  - test100 at target FAR 5%:
    - AUC `0.9273 +/- 0.0047`
    - EER `0.1511 +/- 0.0064`
    - FRR@5%FAR `0.2989 +/- 0.0250`
    - open-set ACC `0.8618 +/- 0.0066`
    - keyword ACC `0.8615 +/- 0.0093`
    - F1 `0.7745 +/- 0.0087`
- Launched extra eval-only tmux session for the same best checkpoint at target FAR 1%:
  - session: `kws_top500_edgespot_eval1far`
  - command uses:
    `TARGET_FAR=0.01 RUN_ID=top500_full_recheck_e20_ep200_far1 bash server/eval_top500_edgespot_best.sh`
  - log:
    `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_far1_edgespot_best_eval.log`
  - purpose: obtain ACC@1%FAR-compatible final dev30/test100 result for Top500 EdgeSpot hybrid.

Colab package:

- Created upload-ready zip with POSIX `/` paths:
  - `D:\Downloads\DoAnTotNghiep\DoAnTotNghiep_code_colab_20260606_FIXED.zip`
- Verified the zip contains:
  - `colab/run_mswc_heavy_flac_train.sh`
  - `data/estimate_mswc_cap.py`
  - `data/convert_opus_to_flac.py`
  - `data/download_mswc.py`
  - `scripts/train.py`
  - `scripts/evaluate.py`

## 2026-06-05 08:45 ICT Update - Colab Heavy FLAC Cap220 Completed

User pasted the Colab log for:

- run id: `colab_mswc_heavy_flac_target6000000_20260604_171246`
- Drive output directory:
  - `/content/drive/MyDrive/DoAnTotNghiep_colab_runs/colab_mswc_heavy_flac_target6000000_20260604_171246`
- runtime:
  - A100-SXM4 40GB
  - torch `2.11.0+cu128`
  - `/content` disk after FLAC conversion: `110G used / 236G`, `47%`

Important data correction:

- The intended target was `TARGET_FILES=6000000`, but the estimator range was
  only `MIN_CAP=180`, `MAX_CAP=220`, `CAP_STEP=20`.
- Selected cap was therefore `MSWC_MAX_PER_WORD=220`, but `hit_target=false`.
- Actual extracted/trained FLAC manifest:
  - train: `2,012,579` files
  - val: `37,138` files
  - total: `2,049,717` files
- Report this run as `MSWC cap220 FLAC`, not as a 6M-clip run.
- To target about 6M clips, rerun only after widening the cap estimator range
  substantially, for example by setting a much larger `MAX_CAP` and estimating
  from metadata before extraction.

Colab stages completed:

1. `DSCNN-L + PCEN + GE2E`
   - run tag:
     - `dscnn_pcen_ge2e_cap220_flac_e15_ep800_colab_mswc_heavy_flac_target6000000_20260604_171246`
   - training:
     - `15 epochs x 800 episodes`
     - best GSC-dev checkpoint-selection metric:
       - `ACC@1%FAR=0.8508` at epoch 9
   - final GSC-dev30:
     - AUC `0.9385 +/- 0.0048`
     - EER `0.1288 +/- 0.0083`
     - FRR@5%FAR `0.2371 +/- 0.0176`
     - open-set ACC `0.8805 +/- 0.0056`
     - keyword ACC `0.8879 +/- 0.0088`
     - F1 `0.8052 +/- 0.0116`
   - final GSC-test100:
     - AUC `0.9387 +/- 0.0047`
     - EER `0.1278 +/- 0.0080`
     - FRR@5%FAR `0.2366 +/- 0.0223`
     - open-set ACC `0.8823 +/- 0.0068`
     - keyword ACC `0.8982 +/- 0.0116`
     - F1 `0.8067 +/- 0.0112`

2. `EdgeSpotFull T4 + PCEN + GE2E`
   - run tag:
     - `edgespot_full_t4_pcen_ge2e_cap220_flac_e15_ep800_colab_mswc_heavy_flac_target6000000_20260604_171246`
   - training:
     - `15 epochs x 800 episodes`
     - best GSC-dev checkpoint-selection metric:
       - `ACC@1%FAR=0.8287` at epoch 9
   - final GSC-dev30:
     - AUC `0.9230 +/- 0.0054`
     - EER `0.1551 +/- 0.0079`
     - FRR@5%FAR `0.3114 +/- 0.0235`
     - open-set ACC `0.8556 +/- 0.0072`
     - keyword ACC `0.8707 +/- 0.0097`
     - F1 `0.7690 +/- 0.0107`
   - final GSC-test100:
     - AUC `0.9131 +/- 0.0062`
     - EER `0.1647 +/- 0.0078`
     - FRR@5%FAR `0.3075 +/- 0.0244`
     - open-set ACC `0.8603 +/- 0.0070`
     - keyword ACC `0.8829 +/- 0.0106`
     - F1 `0.7561 +/- 0.0106`

Interpretation:

- The Colab cap220 FLAC run strongly confirms the current shortlist:
  - `DSCNN-L + PCEN + GE2E` is the higher-accuracy model.
  - `EdgeSpotFull T4 + PCEN + GE2E` remains the compact edge/deployment model.
- On GSC-test100, DSCNN-L beats EdgeSpotFull T4 by:
  - `+2.20 pp` open-set ACC
  - `+2.56 pp` AUC
  - `-3.69 pp` EER
  - `+5.06 pp` F1
- The pasted log ends at `Final artifact sync`; it proves training/evaluation
  completed, but it does not include a final line proving Drive sync completed.
  If needed, verify the Drive folder contains checkpoints/results/logs.

Minor logging issue observed:

- The split summary section printed `words_with_audio=0` and
  `audio_files=0` even though the FLAC manifests were written and training
  loaded them successfully. This is a summary/reporting issue in the folder
  scan path, likely because that summary path is not counting `.flac`; it did
  not affect the actual training/evaluation.

## 2026-06-04 22:45 ICT Update - Heavy MSWC FLAC Runner Targeting About 6M Clips

User clarified that cap50/cap100 is too small and requested implementation of
the plan for a larger MSWC run around 6M clips without returning to full-WAV.

Actions completed:

- Added cap estimator:
  - `data/estimate_mswc_cap.py`
- Added OPUS -> FLAC converter:
  - `data/convert_opus_to_flac.py`
- Added heavy Colab runner:
  - `colab/run_mswc_heavy_flac_train.sh`
- Updated Colab runbook:
  - `docs/colab/mswc_capped_and_flac_runbook_vi.md`
- Updated deprecated full-WAV runbook warning:
  - `docs/colab/full_mswc_24h_runbook_vi.md`

Heavy runner behavior:

- Assumes a fresh Colab runtime.
- Runs `data/download_mswc.py --splits-only` first to get metadata/splits.
- If `MSWC_MAX_PER_WORD` is not manually set, estimates the smallest cap in
  `[180, 200, 220]` that reaches `TARGET_FILES=6000000`.
- Extracts the selected capped MSWC subset.
- Converts OPUS -> FLAC with `data/convert_opus_to_flac.py` and deletes OPUS.
- Builds manifests:
  - `train_files_cap<N>_flac.json`
  - `val_files_cap<N>_flac.json`
- Trains/evaluates:
  - `DSCNN-L + PCEN + GE2E`
  - `EdgeSpotFull T4 + PCEN + GE2E`
- Uses default schedule:
  - `RUN_EPOCHS=15`
  - `RUN_EPISODES=800`
  - `30 classes x 10 samples`
  - final `GSC-dev30` and `GSC-test100`
- Syncs only artifacts to Drive.
- Aborts before training if `/content` exceeds `MAX_DISK_USE_PERCENT=90`.

Recommended Colab command for the user's "near 6M clips" goal:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_heavy_flac_train.sh
TARGET_FILES=6000000 MIN_CAP=180 MAX_CAP=220 CAP_STEP=20 RUN_EPOCHS=15 RUN_EPISODES=800 RUN_DSCNN=1 RUN_EDGESPOT=1 CONVERT_WORKERS=12 CONVERT_BATCH_SIZE=16 FLAC_COMPRESSION_LEVEL=3 bash colab/run_mswc_heavy_flac_train.sh
```

Important:

- `max_per_word=180` means up to 180 clips per word, not 180 words.
- Report this as `MSWC cap≈6M / FLAC`, not full all-clips.
- Do not use `colab/run_full_mswc_24h.sh` for this.

## 2026-06-04 21:55 ICT Update - Implemented Safe Colab MSWC Plan

User requested implementation of the optimized Colab plan after the failed
Full MSWC all-WAV run.

Actions completed:

- Added capped Colab training runner:
  - `colab/run_mswc_capped_train.sh`
- Added long-term FLAC cache shard builder:
  - `colab/build_mswc_flac_shards.sh`
- Added safe Colab runbook:
  - `docs/colab/mswc_capped_and_flac_runbook_vi.md`
- Marked old full all-WAV Colab runbook as unsafe/deprecated for 236GB local
  disk:
  - `docs/colab/full_mswc_24h_runbook_vi.md`

Recommended immediate Colab command:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_capped_train.sh
MSWC_MAX_PER_WORD=50 RUN_EPOCHS=10 RUN_EPISODES=200 RUN_DSCNN=1 RUN_EDGESPOT=1 RUN_EDGESPOT_HYBRID=0 bash colab/run_mswc_capped_train.sh
```

If cap50 completes and disk remains safe, rerun with `MSWC_MAX_PER_WORD=100`.

FLAC cache command for a fresh runtime/session:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/build_mswc_flac_shards.sh
PREPARE_FULL_OPUS=1 SHARD_COUNT=2 SHARD_INDEX=0 CONVERT_WORKERS=12 bash colab/build_mswc_flac_shards.sh
```

Run `SHARD_INDEX=1` in a second session/runtime for the other half. The FLAC
script writes tar files to Drive under
`DoAnTotNghiep_colab_runs/audio_cache/flac_shards/`; it does not write millions
of audio files directly to Drive.

Important:

- Do not rerun `colab/run_full_mswc_24h.sh` on Colab 236GB.
- Do not copy `data/mswc_en/clips` to Drive.
- Treat cap50/cap100 as ablation/shortlist evidence, not final full all-clips
  evidence.

## 2026-06-04 13:20 ICT Update - Colab Full-WAV Failure And Capped Runner

User attempted `colab/run_full_mswc_24h.sh` on Colab Pro+ A100 with about
236GB local `/content` disk. The run reached Full MSWC conversion but local
disk filled. User-reported Colab measurements:

- `data/mswc_en/clips`: `183G`
- WAV files: `6,350,474`, `172.60G`
- OPUS files remaining: `706,560`, `4.24G`
- FLAC files: `0`

Conclusion:

- Full MSWC all-clips converted to WAV is not feasible on a 236GB Colab local
  disk.
- The issue is local `/content` disk, not Google Drive storage.
- Do not rerun `colab/run_full_mswc_24h.sh` for full all-clips WAV on Colab
  unless local disk is much larger.

New action completed:

- Added safe Colab runner:
  - `colab/run_mswc_capped_train.sh`

Runner behavior:

- Uses `data/download_mswc.py --max-per-word "$MSWC_MAX_PER_WORD"` from the
  start, so it extracts only capped clips per word.
- Default cap is `MSWC_MAX_PER_WORD=50`.
- Converts only the capped subset OPUS -> WAV with `--delete-opus`.
- Builds `train_files_cap{N}.json` and `val_files_cap{N}.json`.
- Trains/evaluates the shortlist:
  - `DSCNN-L + PCEN + GE2E`
  - `EdgeSpotFull T4 + PCEN + GE2E`
  - optional `EdgeSpotFull T4 + PCEN + SCAF+GE2E`
- Syncs only small artifacts to Drive:
  - checkpoints, results, reports, logs_colab, configs, colab
- Does not copy audio clips to Drive.

Packaging:

- Regenerated:
  - `D:\Downloads\DoAnTotNghiep\DoAnTotNghiep_code_colab_FIXED.zip`
- Verified the zip includes:
  - `colab/run_mswc_capped_train.sh`
  - `data/download_gsc.py`
  - `data/download_mswc.py`
  - `data/convert_opus.py`
  - `data/build_mswc_file_splits.py`

Recommended next Colab plan:

1. Upload the new `DoAnTotNghiep_code_colab_FIXED.zip` to Drive/Colab.
2. Use `colab/run_mswc_capped_train.sh`, not the old full runner.
3. Start with `MSWC_MAX_PER_WORD=50`; use `100` only if disk remains safe.
4. Keep A100 if available; changing to L4 does not solve local disk limits.

## 2026-06-04 11:00 ICT Update - Completed Vietnamese Thesis Draft

User requested continuing the thesis from local `D:\Downloads\Đồ Án.docx`,
using `D:\Downloads\Đồ án (1).docx` as a reference thesis and
`docs/references/M1-Phan_Thanh_Binh-KWS_Master.pdf` as the KWS protocol
reference.

Actions completed:

- Created reproducible generator:
  - `scripts/make_completed_thesis_vi.py`
- Generated thesis draft outputs:
  - `docs/thesis/Do_An_KWS_completed_vi_2026_06_04.md`
  - `docs/thesis/Do_An_KWS_completed_vi_2026_06_04.docx`
- Verification:
  - DOCX has 146 paragraphs, 4 tables, and 3 embedded result figures.
  - Original user Word file under `D:\Downloads` was not overwritten.

Included content:

- Vietnamese thesis continuation/completion based on the user's skeleton:
  Acknowledgements, Introduction, Dataset, Model Architecture, System Pipeline,
  Experimental Results and Discussion, Demo System, Conclusion, Limitations,
  Future Work.
- Tables for Microset, Full MSWC 16-pipeline ablation, Full MSWC shortlist
  manifest20/manifest50, and Top500 recheck.
- Metric/protocol section based on the M1 KWS reference style: FAR, FRR,
  ACC@1%FAR, ACC@5%FAR, AUC, EER, DET curve, and GSC-test100.

## 2026-06-04 11:15 ICT Update - Colab Full MSWC 24h Runner

User wants to use Colab Pro+ units for a continuous run, prioritizing Full MSWC
first, with continuous checkpoint/evaluation artifact saving. User explicitly
does not want WAV clips saved to Google Drive because that costs too much time
and unit.

Actions completed:

- Added Colab runner:
  - `colab/run_full_mswc_24h.sh`
- Added Colab runbook:
  - `docs/colab/full_mswc_24h_runbook_vi.md`
- Added Windows packaging helper:
  - `colab/package_code_for_colab.ps1`

Runner behavior:

- Does not call `data/mswc_drive_cache.py`.
- Does not copy `data/mswc_en/clips` or any WAV cache to Drive.
- Keeps Full MSWC audio local under `/content`.
- Continuously syncs only small/important artifacts to Drive:
  - `checkpoints/`
  - `results/`
  - `reports/`
  - `logs_colab/`
  - `configs/`
- Preflight disk check:
  - exits if `/content` free space is under 120GB;
  - warns if under 150GB.
- Default priority:
  1. Full MSWC all-clips `DSCNN-L + PCEN + GE2E`.
  2. Full MSWC all-clips `EdgeSpotFull T4 + PCEN + GE2E`.
  3. If time remains, `EdgeSpotFull T4 + PCEN + SCAF+GE2E`.
  4. If time remains, Top500 follow-up.

Recommended Colab GPU guidance captured in runbook:

- `A100` is the best balanced choice for a 24h run with 490 units.
- `H100` is fastest for training but wastes value during the data-download
  phase and burns units faster.
- `L4` is cheaper/slower; likely fewer completed stages.
- `T4` is not recommended for Full MSWC all-clips.

## 2026-06-04 12:35 ICT Update - Thesis Intro Vietnamese Draft

User requested Vietnamese thesis content and English-writing guidance, referencing:

- Google Docs thesis/reference links supplied by user.
- Local PDF reference: `docs/references/M1-Phan_Thanh_Binh-KWS_Master.pdf`.

Actions completed:

- Read the orchestration/init/state/handoff files.
- Extracted the reference PDF structure with bundled Python `pypdf`.
  - The thesis sample structure is: Declaration, Acknowledgements, Abstract,
    Chapter 1 Introduction with Context and Motivation/Objectives/Report
    Structure, then Background, Methodology, Results & Discussion, Conclusions.
- Google Docs links could not be read through the available tool context; user
  should enable "Anyone with the link can view" or paste/export the current text
  if exact Google Doc editing is needed.
- Created local draft:
  - `docs/thesis/thesis_intro_vi_guidance_2026_06_04.md`

Claim guidance captured in the draft:

- Microset is the main controlled evidence for architecture selection:
  - EdgeSpotFull T4 + PCEN + SCAF+GE2E test100:
    - `ACC@5%FAR=86.12%`
    - `AUC=95.61%`
    - `EER=11.54%`
    - `F1=82.41%`
- Full MSWC phase-1 is an ablation/screening result, not final training:
  - DSCNN-L + PCEN + GE2E is best in the 16-pipeline screening:
    - `ACC@1%FAR=76.67%`
  - EdgeSpotFull T4 + PCEN + GE2E is best within EdgeSpot group:
    - `ACC@1%FAR=72.94%`
- Full MSWC shortlist manifest20 is current accuracy shortlist evidence:
  - DSCNN-L + PCEN + GE2E test100:
    - `ACC@1%FAR=82.10 +/- 0.87`
    - `ACC@5%FAR=86.05 +/- 0.66`
  - EdgeSpotFull T4 + PCEN + GE2E test100:
    - `ACC@1%FAR=79.58 +/- 0.91`
    - `ACC@5%FAR=83.06 +/- 0.82`
- Top500 epoch13 is reproducible local artifact evidence; epoch25 remains
  historical/log-only unless the real checkpoint artifact is found.

## 2026-06-02 15:50 ICT Update - Top500Full Recheck Launched

User requested a Top500Full recheck to compare the old EdgeSpot hybrid result
against `DSCNN-L + PCEN + GE2E`.

Local/server code added:

- `data/build_mswc_top500_profile.py`
  - creates a separate `data/mswc_top500_full` profile;
  - preserves current Full MSWC split/manifests under `data/mswc_en`;
  - uses the legacy Top500 split from `data/download_mswc.py`:
    - top 500 words by English MSWC metadata count;
    - first 450 words for train;
    - next 50 words for validation;
    - no eval words by default.
- `data/build_mswc_file_splits.py`
  - optimized extracted-clip manifest building with `os.scandir` instead of
    slower `Path.iterdir()+is_file()` on large NFS word folders.
- `server/run_top500_full_recheck.sh`
  - builds/reuses the clean Top500Full profile;
  - re-evaluates local artifact `epoch_13.pt` for the old EdgeSpot hybrid;
  - trains/evaluates `DSCNN-L + PCEN + GE2E`;
  - trains/evaluates `EdgeSpotFull T4 + PCEN + SCAF+GE2E` from scratch for a
    same-schedule comparison.
- `server/launch_top500_recheck.sh`
  - starts tmux session `kws_top500_recheck`.

Verified before launch:

- Local:
  - `python -m py_compile data\build_mswc_file_splits.py data\build_mswc_top500_profile.py`
  - `python -m pytest tests\test_build_mswc_file_splits.py tests\test_build_mswc_top500_profile.py -q --basetemp .codex_tmp\pytest -p no:cacheprovider` -> `4 passed`.
- Server ict6:
  - `python -m py_compile data/build_mswc_file_splits.py data/build_mswc_top500_profile.py`
  - `python -m pytest tests/test_build_mswc_file_splits.py tests/test_build_mswc_top500_profile.py -q` -> `4 passed`.
- Uploaded available legacy checkpoint:
  - `checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt`
  - size on server: `2.8M`.

Active server state:

- tmux session: `kws_top500_recheck`
- run ID: `top500_full_recheck_e20_ep200`
- log: `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200.log`
- summary TSV: `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_runs.tsv`
- data profile: `data/mswc_top500_full`
- source clips: `data/mswc_en/clips`
- GPU: `4`
- train schedule:
  - `epochs=20`
  - `episodes=200`
  - `n_classes=30`
  - `n_samples=10`
  - `max_per_word=0` (Top500 full clips, not capped)
  - final eval: dev30 + test100
  - target FAR for top-level metric: `0.05`; JSON also contains ACC@1%FAR and ACC@5%FAR.

Observed launch state:

- First launch at `15:35` failed immediately due `ModuleNotFoundError: No module named 'data'` in the new profile script.
- This was fixed by adding project-root `sys.path` in `data/build_mswc_top500_profile.py`.
- Relaunched at `15:42:54`.
- Current stage at latest check:
  - building clean Top500Full manifest;
  - split confirmed in log: `450 train, 50 val, 485 eval` from legacy split function;
  - eval words are intentionally not used;
  - after optimizing `data/build_mswc_file_splits.py`, progress improved from an initial
    `2/500` with a long estimate to `31/500` after about `6m18s`;
  - current manifest ETA is roughly under 1 hour, not 5 hours.
- GPU 4 is expected to stay idle during manifest building; training starts only after `train_files.json` and `val_files.json` are written.

Experiment order:

1. Re-evaluate old available EdgeSpot hybrid checkpoint:
   - `EdgeSpotFull T4 + PCEN + SCAF+GE2E`
   - checkpoint: `checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt`
   - purpose: check whether old Top500 epoch13 accuracy reproduces.
2. Train/evaluate:
   - `DSCNN-L + PCEN + GE2E`
   - run tag: `dscnn_pcen_ge2e_top500_full_recheck_e20_ep200`
   - purpose: check whether the newer DSCNN+PCEN+GE2E candidate beats old Top500 EdgeSpot hybrid.
3. Train/evaluate from scratch under same schedule:
   - `EdgeSpotFull T4 + PCEN + SCAF+GE2E`
   - run tag: `edgespot_full_t4_pcen_scaf_ge2e_top500_full_recheck_e20_ep200`
   - purpose: apples-to-apples Top500 schedule comparison.

Monitoring commands:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux attach -t kws_top500_recheck
tail -n 120 /storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200.log
cat /storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_runs.tsv
ls -lh /storage/<user>/an_kws/DoAnTotNghiep/data/mswc_top500_full/splits
```

Claim hygiene:

- Do not cite any new Top500 recheck result until the summary TSV row has
  `dev_status=ok` and/or `test_status=ok`.
- The old epoch25 Top500 result is still historical/log-only unless a real
  `epoch_25.pt` checkpoint is found.
- Current reproducibility checkpoint is epoch13 only.

## 2026-06-02 18:10 ICT Update - Top500 Recheck Progress

Verified on ict6 from `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200.log`
and copied JSON evidence under `reports/top500_full_recheck/raw/`.

Completed:

- Legacy checkpoint re-evaluation completed:
  - run tag: `edgespot_full_t4_pcen_scaf_ge2e_top500_epoch13_reval`
  - checkpoint: `checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt`
  - summary TSV row: `evaluate_existing ... dev_status=ok test_status=ok`

Metrics:

| Split | Runs | Target FAR | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev30 | 30 | 5% | 86.68 | 88.88 | 95.12 | 12.03 | 81.71 |
| test100 | 100 | 5% | 85.62 | 88.79 | 95.34 | 11.51 | 82.45 |

Interpretation:

- The dev30 recheck reproduces the old Top500 epoch13 result:
  old note was about `ACC@5%FAR=88.87%`; current recheck is `88.88%`.
- This confirms the old epoch13 artifact is stable/reproducible.
- This does not confirm epoch25 because `epoch_25.pt` is still missing.

Active:

- `DSCNN-L + PCEN + GE2E` is training:
  - run tag: `dscnn_pcen_ge2e_top500_full_recheck_e20_ep200`
  - latest observed: epoch `12/20` running.
  - best checkpoint-selection metric so far at epoch 10:
    - `GSC-dev ACC@1%FAR=79.20`
    - `GSC-dev ACC@5%FAR=83.39`
- These are interim 5-run checkpoint-selection metrics, not final dev30/test100.
- After DSCNN finishes, the runner will evaluate dev30/test100, then train/evaluate
  `EdgeSpotFull T4 + PCEN + SCAF+GE2E` from scratch under the same schedule.

## 2026-06-02 15:12 ICT Update - Manifest50 Shortlist Completed

Verified live on ict6 at `2026-06-02 15:12 +07`.

Server state:

- No active `scripts/train.py` or `scripts/evaluate.py` process for the KWS max50 job.
- `kws_manifest50_fixed` and `kws_dscnn_max50_recovery` tmux sessions are no longer listed.
- GPU 4 is idle: `11 MiB`, `0%` utilization at check time.
- Main TSV:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv`
  - EdgeSpot row: `train_status=ok`, `dev30_status=ok`, `test100_status=ok`.
  - Original DSCNN row remains `failed_rc_1` because it failed before recovery.
- Recovery TSV:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.tsv`
  - DSCNN recovery row: `train_status=ok`, `dev30_status=ok`, `test100_status=ok`.

Local evidence copied:

- `reports/full_mswc_shortlist_manifest50/logs/`
- `reports/full_mswc_shortlist_manifest50/raw/full_mswc_shortlist_manifest50_clips_e20_ep200/`
- `reports/full_mswc_shortlist_manifest50/shortlist_results_summary.md`
- `reports/full_mswc_shortlist_manifest50/shortlist_results_summary.csv`

Manifest50 test100 metrics:

| Pipeline | Params | ACC@1%FAR | ACC@5%FAR | AUC | EER | FRR@1%FAR | FRR@5%FAR | KW-ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | 412,900 | 80.96 +/- 1.16 | 84.68 +/- 0.70 | 90.45 +/- 0.68 | 17.42 +/- 1.08 | 59.25 +/- 3.99 | 35.95 +/- 2.46 | 87.63 +/- 1.25 | 74.34 +/- 1.43 |
| EdgeSpotFull T4 + PCEN + GE2E | 130,594 | 77.14 +/- 0.89 | 82.24 +/- 0.74 | 87.74 +/- 0.66 | 20.19 +/- 0.90 | 71.20 +/- 3.12 | 42.02 +/- 2.53 | 83.49 +/- 1.22 | 70.73 +/- 1.16 |

Interpretation:

- DSCNN-L remains the higher-accuracy candidate.
- EdgeSpotFull T4 remains the compact deployment candidate.
- Max50 is harder than manifest20 with the current 20-epoch schedule:
  - DSCNN ACC@1%FAR: `82.10` -> `80.96`.
  - EdgeSpot ACC@1%FAR: `79.58` -> `77.14`.
- Do not claim max50 improves accuracy. Claim it is a robustness follow-up that confirms the same shortlist ranking and indicates longer/tuned training is needed if using more clips per word.

## 2026-06-02 10:37 ICT Update - Manifest50 EdgeSpot Running Test100, DSCNN Recovery Queued

Verified on ict6 at `2026-06-02 10:37 +07`.

Manifest50 build is complete and valid:

- `data/mswc_en/splits/train_files_max50.json`: `939,108` files.
- `data/mswc_en/splits/val_files_max50.json`: `18,598` files.
- `data/mswc_en/splits/file_manifest_summary_max50.json`.
- Summary mode: `source=clips`, `max_per_word=50`, `train_words=37387`, `val_words=763`, no missing train/val words.
- Some `short_*_words` are expected because those words have fewer than 50 clips.

Server job state:

- Main tmux session: `kws_manifest50_fixed`.
- Recovery tmux session: `kws_dscnn_max50_recovery`.
- Main run ID: `full_mswc_shortlist_manifest50_clips_e20_ep200`.
- Main TSV: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv`.
- Recovery TSV: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.tsv`.
- Recovery log: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.log`.

EdgeSpotFull T4 + PCEN + GE2E max50:

- Train completed OK at `2026-06-02 10:28`.
- Best checkpoint: `checkpoints/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest50_clips_e20_ep200/best.pt`.
- Best GSC-dev checkpoint-selection metric during train: `ACC@1%FAR=0.7727` at epoch 20.
- `dev30` completed and `test100` was running at last check:
  - process: `python scripts/evaluate.py ... --gsc-query-split test --n-runs 100 ... --edge-tau 4`.
- Do not cite final EdgeSpot test100 max50 until TSV row shows `train_status=ok`, `dev30_status=ok`, `test100_status=ok`.

DSCNN-L + PCEN + GE2E max50:

- Initial train failed during epoch 14 due audio/NFS read error:
  - `soundfile.LibsndfileError: Error opening 'data/mswc_en/clips/frampton/common_voice_en_20099925.opus': System error.`
- Valid checkpoints exist:
  - `checkpoints/dscnn_pcen_ge2e_full_mswc_shortlist_manifest50_clips_e20_ep200/latest.pt`
  - `checkpoints/dscnn_pcen_ge2e_full_mswc_shortlist_manifest50_clips_e20_ep200/best.pt`
- Last confirmed best metric before failure: `GSC-dev ACC@1%FAR=0.7850` at epoch 10.
- Recovery script now queued:
  - `server/run_manifest50_dscnn_recovery.sh`
  - waits for EdgeSpot final summary row;
  - resumes DSCNN from `latest.pt`;
  - passes `--initial-best-metric 0.7850`;
  - trains to epoch 20;
  - runs `dev30` and `test100`;
  - writes recovery TSV separately.

Code fix applied for long MSWC robustness:

- `src/data/mswc_dataset.py`
  - builds per-label sample indices;
  - retries several same-label samples when `load_waveform` fails;
  - falls back to silence only if all retry attempts fail.
- `tests/test_mswc_microset.py`
  - adds coverage for unreadable manifest audio retry.

Verified:

- Local: `python -m py_compile src\data\mswc_dataset.py`.
- Local: `python -m pytest tests\test_mswc_microset.py tests\test_build_mswc_file_splits.py -q --basetemp .codex_tmp\pytest -p no:cacheprovider` -> `8 passed`.
- Server ict6: `python -m py_compile src/data/mswc_dataset.py`.
- Server ict6: `python -m pytest tests/test_mswc_microset.py tests/test_build_mswc_file_splits.py -q` -> `8 passed`.

Next check command:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tail -n 120 /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200.log
tail -n 120 /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.log
cat /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv
cat /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_dscnn_recovery.tsv
```

## 2026-06-01 23:36 ICT Update - Max50 Follow-Up Fixed And Relaunched

Main report:

- `D:\Downloads\DoAnTotNghiep\docs\reports\optimization_report_2026_06_01_vi.md`

Code/server fixes applied for the broken max50 follow-up:

- `data/build_mswc_file_splits.py`
  - supports `--source clips`;
  - accepts `.wav`, `.opus`, `.flac`, `.mp3`;
  - for capped manifests, stops reading each word folder after collecting `max_per_word` audio files instead of scanning/sorting every file in very large folders.
- `server/run_full_mswc_shortlist_manifest50.sh`
  - builds `train_files_max50.json` / `val_files_max50.json` from extracted clips.
- `server/launch_shortlist_manifest50_fixed.sh`
  - launches tmux session `kws_manifest50_fixed`;
  - sets `RUN_ID=MATRIX_ID=full_mswc_shortlist_manifest50_clips_e20_ep200`.
- `tests/test_build_mswc_file_splits.py`
  - includes extracted clips and `.opus` coverage.

Verified:

- Local: `python -m py_compile data\build_mswc_file_splits.py`
- Local: `python -m pytest tests\test_build_mswc_file_splits.py -q` -> `3 passed`
- Server ict6: `python -m py_compile data/build_mswc_file_splits.py`
- Server ict6: `python -m pytest tests/test_build_mswc_file_splits.py -q` -> `3 passed`

Active server job:

- tmux session: `kws_manifest50_fixed`
- host path: `/storage/<user>/an_kws/DoAnTotNghiep`
- GPU: `4`
- current stage at last observation: building max50 manifests from extracted clips
- process observed:
  - `python data/build_mswc_file_splits.py --data-dir data/mswc_en --max-per-word 50 --output-suffix max50 --source clips`
- logs:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_wait_gpu.log`
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_bootstrap.log`
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200.log` once training starts
- expected summary TSV after train/eval:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_clips_e20_ep200_runs.tsv`

Expected runtime from relaunch around 2026-06-01 23:31 ICT:

- manifest build: roughly 45-90 minutes, depending on NFS;
- full two-pipeline train/eval: roughly 14-22 hours total;
- do not cite max50 results until the runs TSV has 2 rows with `train_status=ok`, `dev30_status=ok`, and `test100_status=ok`.

## Latest Shortlist Result State

Verified live on ict6 at 2026-06-01 15:36 ICT:

- tmux/process state:
  - no active `kws_shortlist_manifest20` tmux session was listed.
  - no active `scripts/train.py`, `scripts/evaluate.py`, `run_full_mswc_shortlist`, or `launch_shortlist` process was observed.
- Server summary:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200_runs.tsv`
  - contains 2 rows, both `train_status=ok`, `dev30_status=ok`, `test100_status=ok`.
- Local copied evidence:
  - `D:\Downloads\DoAnTotNghiep\reports\full_mswc_shortlist_manifest20\raw\full_mswc_shortlist_manifest20_e20_ep200_runs.tsv`
  - `D:\Downloads\DoAnTotNghiep\reports\full_mswc_shortlist_manifest20\shortlist_results_summary.md`
  - `D:\Downloads\DoAnTotNghiep\reports\full_mswc_shortlist_manifest20\shortlist_results_summary.csv`
  - DET curves and JSONs copied under `reports\full_mswc_shortlist_manifest20\raw\full_mswc_shortlist_manifest20_e20_ep200\...`

Final shortlist test100 metrics:

| Pipeline | Runs | ACC@1%FAR | FRR@1%FAR | ACC@5%FAR | FRR@5%FAR | AUC | EER | KW-ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + PCEN + GE2E | 100 | 82.10 +/- 0.87 | 55.48 +/- 2.94 | 86.05 +/- 0.66 | 31.38 +/- 2.28 | 91.57 +/- 0.58 | 16.25 +/- 0.86 | 88.90 +/- 1.25 | 75.90 +/- 1.16 |
| EdgeSpotFull T4 + PCEN + GE2E | 100 | 79.58 +/- 0.91 | 63.45 +/- 3.18 | 83.06 +/- 0.82 | 40.01 +/- 2.96 | 87.22 +/- 0.75 | 20.40 +/- 1.01 | 83.01 +/- 1.49 | 70.46 +/- 1.30 |

Interpretation:

- Accuracy candidate: `DSCNN-L + PCEN + GE2E`.
- Edge/device candidate: `EdgeSpotFull T4 + PCEN + GE2E`.
- DSCNN-L leads EdgeSpotFull T4 by `+2.52 pp` ACC@1%FAR and `+5.44 pp` F1 on test100.
- Do not launch another broad 16-combo matrix. The next experimental step should be targeted: improve EdgeSpotFull T4 if compact edge deployment remains the thesis target, or lock DSCNN-L if highest accuracy is the primary target.

## Active Follow-Up State

Checked again on ict6 at 2026-06-01 23:03 ICT:

- `kws_manifest50` is not active.
- No real `data/build_mswc_file_splits.py` or `scripts/train.py` process was observed.
- No max50 manifest files were present:
  - `data/mswc_en/splits/train_files_max50.json`
  - `data/mswc_en/splits/val_files_max50.json`
  - `data/mswc_en/splits/file_manifest_summary_max50.json`
- No max50 summary TSV was present:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_e20_ep200_runs.tsv`
- Bootstrap log exists and was last modified at `2026-06-01 16:03:00 +0700`:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_e20_ep200_bootstrap.log`
- Conclusion: the max50 follow-up stopped during manifest construction before producing usable output. Do not cite max50 as running or completed.

Previous attempted launch on ict6 at 2026-06-01 15:58 ICT:

- tmux session: `kws_manifest50`
- Bootstrap log: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_e20_ep200_bootstrap.log`
- Main runner log after manifest build: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_e20_ep200.log`
- Summary TSV after training/eval starts: `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest50_e20_ep200_runs.tsv`
- Script: `/storage/<user>/an_kws/DoAnTotNghiep/server/run_full_mswc_shortlist_manifest50.sh`
- Purpose: create named max50 manifests, then train/evaluate only the two shortlist candidates:
  - `DSCNN-L + PCEN + GE2E`
  - `EdgeSpotFull T4 + PCEN + GE2E`
- Manifest files are separate from manifest20:
  - `data/mswc_en/splits/train_files_max50.json`
  - `data/mswc_en/splits/val_files_max50.json`
  - `data/mswc_en/splits/file_manifest_summary_max50.json`
- Do not overwrite `train_files.json` / `val_files.json`; those are the manifest20 evidence for prior results.
- Previous observed state: `python data/build_mswc_file_splits.py --max-per-word 50 --output-suffix max50` was scanning `data/mswc_en/en.tar.gz`.

## Project State

- Workspace: `D:\Downloads\DoAnTotNghiep`
- Existing long handoff: `D:\Downloads\DoAnTotNghiep\docs\session_handoff_2026_05_29.md`
- Root bootstrap files now exist:
  - `D:\Downloads\DoAnTotNghiep\AGENTS.md`
  - `D:\Downloads\DoAnTotNghiep\CLAUDE.md`
  - `D:\Downloads\DoAnTotNghiep\docs\agent_orchestration_init.md`

## Current Research Direction

- The user wants to understand and test model/feature/loss combinations, not only train one final full-MSWC model.
- Candidate experiment matrix should focus on DSCNN and EdgeSpotFull T4, exactly these 12 combinations:
  - DSCNN + MFCC + Triplet
  - DSCNN + MFCC + SCAF
  - DSCNN + MFCC + GE2E
  - DSCNN + MFCC + SCAF+GE2E
  - DSCNN + PCEN + Triplet
  - DSCNN + PCEN + SCAF
  - DSCNN + PCEN + GE2E
  - DSCNN + PCEN + SCAF+GE2E
  - EdgeSpotFull T4 + PCEN + Triplet
  - EdgeSpotFull T4 + PCEN + SCAF
  - EdgeSpotFull T4 + PCEN + GE2E
  - EdgeSpotFull T4 + PCEN + SCAF+GE2E
- BC-ResNet is not requested for this matrix.
- EdgeSpotFull+MFCC is not in the requested matrix; do not add it unless the user asks for that ablation.

## Evidence/Claim State

- Microset is the strongest thesis evidence for selecting EdgeSpotFull T4 + SCAF+GE2E.
- Top500 epoch13 is the local/demo artifact.
- Top500 epoch25 was observed in Colab/logs but should not be overclaimed unless checkpoint artifact is available.
- Open-set UI sampled evaluation is demo-level, not a replacement for `gsc_edgespot_exact test100`.

## Server State

Last observed by Codex on ict6/frontend:

- Host path: `/storage/<user>/an_kws/DoAnTotNghiep`
- Full MSWC English archive downloaded successfully:
  - `data/mswc_en/en.tar.gz`
  - Log: `/storage/<user>/an_kws/logs/kws_full_mswc.log`
- Extraction reached `100%|6662520/6662520` at `2026-05-29 18:12:52`.
- Log line `Extracted 2081233 clips for 38150 words (skipped 1 non-target files)` should not be treated as the total available MSWC English clip count. It is an extraction-run statistic from the resumed/full extraction script, not the final training manifest size and not proof that only 2.08M clips exist.
- Correct MSWC count check on 2026-05-31:
  - metadata words: `38174`
  - metadata total clips: `6624343`
  - split metadata clip sum:
    - train words: `6527593`
    - val words: `96750`
    - total train+val: `6624343`
  - sample actual on-disk counts matched metadata:
    - `yes`: metadata `3316`, actual files `3316`
    - `stop`: metadata `2398`, actual files `2398`
    - `marvin`: metadata `36`, actual files `36`
    - `zvooq`: metadata `10`, actual files `10`
- Use this wording: MSWC English metadata/split covers `6.62M` clips across `38,174` words; current experiments train from a capped manifest of `527,069` train files and `10,637` val files (`max_per_word=20`).
- The old download PID `2962205` was stopped after extraction because it was stuck in an expensive final recursive OPUS/WAV count on NFS.
- Downloader patch now skips that recursive count by default and exposes `--exact-final-count`.

MSWC manifest state:

- Script: `data/build_mswc_file_splits.py`
- Log: `/storage/<user>/an_kws/logs/kws_manifest_max20.log`
- Outputs:
  - `data/mswc_en/splits/train_files.json`
  - `data/mswc_en/splits/val_files.json`
  - `data/mswc_en/splits/file_manifest_summary.json`
- Observed counts:
  - `train_files.json`: 527,069 files
  - `val_files.json`: 10,637 files
- Manifest cap: 20 files/word.

Completed smoke matrix:

- tmux session: `kws_matrix12`
- Log: `/storage/<user>/an_kws/logs/full_mswc_12_combo_manifest20_smoke.log`
- Summary: `/storage/<user>/an_kws/logs/full_mswc_12_combo_manifest20_smoke_runs.tsv`
- Wrapper: `/storage/<user>/an_kws/run_matrix12_smoke.sh`
- Settings:
  - `MATRIX_ID=full_mswc_12_combo_manifest20_smoke`
  - `MATRIX_EPOCHS=1`
  - `MATRIX_EPISODES=20`
  - `MATRIX_N_CLASSES=20`
  - `MATRIX_N_SAMPLES=10`
  - `MATRIX_MAX_PER_WORD=20`
  - `MATRIX_GSC_RUNS=1`
  - `GPU_ID=4`
- First smoke result observed:
  - `DSCNN + MFCC + Triplet`
  - status: `ok`
  - GSC-dev ACC@1%FAR: `0.6917`
  - GSC-dev ACC@5%FAR: `0.6667`
  - checkpoint directory: `checkpoints/dscnn_mfcc_triplet_full_mswc_12_combo_manifest20_smoke_e1_ep20`
- Matrix was continuing into `DSCNN + MFCC + SCAF` when user said to stop polling and let it run.

Latest verified state on 2026-05-30:

- Smoke matrix `full_mswc_12_combo_manifest20_smoke` finished all 12 requested combinations with `status=ok`.
- Summary file:
  - `/storage/<user>/an_kws/logs/full_mswc_12_combo_manifest20_smoke_runs.tsv`
- Smoke settings:
  - `MATRIX_EPOCHS=1`
  - `MATRIX_EPISODES=20`
  - `MATRIX_N_CLASSES=20`
  - `MATRIX_N_SAMPLES=10`
  - `MATRIX_MAX_PER_WORD=20`
  - `MATRIX_GSC_RUNS=1`
- This smoke result proves the 12-combo pipeline runs end-to-end, but it is not a thesis-quality ranking because the training budget is intentionally tiny.

Active phase-1 training launcher:

- tmux session:
  - `kws_matrix12_phase1_wait`
- Wait log:
  - `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_wait_gpu.log`
- Matrix log, once launched:
  - `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20.log`
- Summary TSV, once launched:
  - `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_runs.tsv`
- Script:
  - `/storage/<user>/an_kws/DoAnTotNghiep/server/wait_gpu_then_run_full_mswc_matrix.sh`
  - `/storage/<user>/an_kws/DoAnTotNghiep/server/run_full_mswc_experiment_matrix.sh`
- Settings:
  - `MATRIX_ID=full_mswc_12_combo_phase1_e5_ep150_manifest20`
  - `MATRIX_EPOCHS=5`
  - `MATRIX_EPISODES=150`
  - `MATRIX_N_CLASSES=30`
  - `MATRIX_N_SAMPLES=10`
  - `MATRIX_MAX_PER_WORD=20`
  - `MATRIX_WORKERS=8`
  - `MATRIX_GSC_RUNS=3`
  - `MATRIX_GSC_EVERY=1`
  - `MATRIX_K_SHOT=10`
  - `GPU_CANDIDATES="4 0 1 6 2 3 5 7"`
  - `GPU_MEMORY_MAX_MB=1500`
  - `GPU_UTIL_MAX=15`
  - `GPU_REQUIRED_IDLE_CHECKS=2`
  - `GPU_POLL_SECONDS=180`
- Last observed wait log line:
  - `No candidate GPU is idle enough yet`
- This job will wait until one candidate GPU has memory <= 1500 MiB and utilization <= 15% for two consecutive checks, then launch the 12-combo matrix on that single GPU.

Observed progress from user-provided log on 2026-05-30:

- GPU wait finished:
  - `Candidate GPU 4 passed idle check 2/2`
  - `Launching matrix on GPU 4`
- Runtime evidence:
  - `torch: 1.12.1+cu102 cuda: 10.2 available: True`
  - `gpu0: Tesla K80`
  - Training data loaded from explicit manifest:
    - `527069 train file paths`
    - `10637 val file paths`
    - `Dataset: 527069 samples, 37387 words`
- Completed phase-1 runs visible in the log:
  - `dscnn_mfcc_triplet_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
    - status: `Finished experiment OK`
    - best GSC-dev ACC@1%FAR: `0.6930`
  - `dscnn_mfcc_scaf_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
    - status: `Finished experiment OK`
    - best GSC-dev ACC@1%FAR: `0.7072`
  - `dscnn_mfcc_ge2e_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
    - status: `Finished experiment OK`
    - best GSC-dev ACC@1%FAR: `0.7230`
  - `dscnn_mfcc_scaf_ge2e_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
    - status: `Finished experiment OK`
    - best GSC-dev ACC@1%FAR: `0.6924`
- Active run visible at end of provided log:
  - `dscnn_pcen_triplet_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
  - epoch 1 training line observed, no final status yet in provided log.
- Interim interpretation:
  - Among the completed MFCC/DSCNN combinations, `DSCNN + MFCC + GE2E` is currently strongest by GSC-dev ACC@1%FAR.
  - This is still phase-1 screening, not final thesis evidence. Wait for all 12 runs and then parse the summary/checkpoints.

Latest verified/resume state on 2026-05-30 11:34 ICT:

- Direct server check showed no active `scripts/train.py` process and no original phase-1 tmux session.
- GPU 4 was idle enough to reuse:
  - `GPU 4: 11 MiB, 0% util` before relaunch.
- Phase-1 TSV had only 6 completed `ok` rows:
  - `dscnn_mfcc_triplet...`
  - `dscnn_mfcc_scaf...`
  - `dscnn_mfcc_ge2e...`
  - `dscnn_mfcc_scaf_ge2e...`
  - `dscnn_pcen_triplet...`
  - `dscnn_pcen_scaf...`
- Old log showed the 7th run stopped after epoch 3/5:
  - `dscnn_pcen_ge2e_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150`
  - last checkpoint: `epoch_03.pt` / `latest.pt`
  - prior best GSC-dev ACC@1%FAR observed in log: `0.7161` at epoch 2.
- Added/verified resume support:
  - `scripts/train.py` now accepts `--initial-best-metric`.
  - `server/wait_gpu_then_run_full_mswc_matrix.sh` now accepts `MATRIX_RUNNER`.
  - `server/run_full_mswc_experiment_matrix_resume.sh` skips completed `ok` rows, resumes partial runs from `latest.pt`, backs up `best.pt` to `pre_resume_best.pt`, and can append EdgeSpotFull T4 + MFCC supplementary ablations.
- Uploaded those 3 files to ict6 and verified:
  - `bash -n server/wait_gpu_then_run_full_mswc_matrix.sh`
  - `bash -n server/run_full_mswc_experiment_matrix_resume.sh`
  - `python -m py_compile scripts/train.py`
  - `python scripts/train.py --help` shows `--initial-best-metric`.
- Relaunched tmux:
  - session: `kws_matrix_phase1_resume`
  - wait log: `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_wait_gpu.log`
  - resume log: `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_resume.log`
  - primary summary: `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_runs.tsv`
  - supplementary EdgeSpot+MFCC summary, if reached: `/storage/<user>/an_kws/logs/full_mswc_edgespot_mfcc_4_combo_phase1_e5_ep150_manifest20_runs.tsv`
- Relaunch command used:
  - `INCLUDE_EDGESPOT_MFCC=1`
  - `MATRIX_ID=full_mswc_12_combo_phase1_e5_ep150_manifest20`
  - `EXTRA_MATRIX_ID=full_mswc_edgespot_mfcc_4_combo_phase1_e5_ep150_manifest20`
  - `MATRIX_EPOCHS=5`, `MATRIX_EPISODES=150`, `MATRIX_N_CLASSES=30`, `MATRIX_N_SAMPLES=10`, `MATRIX_MAX_PER_WORD=20`, `MATRIX_WORKERS=8`, `MATRIX_GSC_RUNS=3`, `MATRIX_K_SHOT=10`.
- Verified after relaunch:
  - GPU 4 passed idle check 2/2 and launched.
  - First 6 completed runs were skipped.
  - `dscnn_pcen_ge2e...` resumed with:
    - `--resume checkpoints/dscnn_pcen_ge2e_full_mswc_12_combo_phase1_e5_ep150_manifest20_e5_ep150/latest.pt`
    - `--initial-best-metric 0.716100`
  - `pre_resume_best.pt` was created before continuing.
  - Active train process observed:
    - `python scripts/train.py ... --loss ge2e ... --resume ... --initial-best-metric 0.716100`
- Resume correction after first relaunch:
  - First resume attempt exposed a server compatibility issue: PyTorch 1.12 rejects `torch.load(..., weights_only=False)`.
  - Patched `scripts/train.py::load_checkpoint()` to try `weights_only=False` and fall back to plain `torch.load(path)` on `TypeError`.
  - Killed the bad `kws_matrix_phase1_resume` session, removed the generated `failed_rc_1` row from the TSV, uploaded the patch, and relaunched the same tmux session.
  - Current verified state after relaunch:
    - session: `kws_matrix_phase1_resume`
    - active process: `python scripts/train.py ... --loss ge2e ... --resume ... --initial-best-metric 0.716100`
    - the run passed checkpoint loading and reached data loading/training setup.
    - TSV is clean with 6 completed `ok` rows; `dscnn_pcen_ge2e...` is currently running and not yet appended.

Final verified state on 2026-05-31 09:22 ICT:

- Host checked: `ictserver6`.
- No active `scripts/train.py`, `run_full_mswc_experiment_matrix_resume.sh`, or wait-gpu process remains.
- The phase-1 12-combo matrix finished successfully:
  - summary: `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_runs.tsv`
  - rows: `12`
  - status counts: `ok=12`
  - final primary run finished: `2026-05-30 17:17:59`
- The supplementary EdgeSpotFull T4 + MFCC 4-combo matrix also finished successfully:
  - summary: `/storage/<user>/an_kws/logs/full_mswc_edgespot_mfcc_4_combo_phase1_e5_ep150_manifest20_runs.tsv`
  - rows: `4`
  - status counts: `ok=4`
  - final supplementary run finished: `2026-05-30 19:23:04`
- The supplementary runs are logged in the shared resume log:
  - `/storage/<user>/an_kws/logs/full_mswc_12_combo_phase1_e5_ep150_manifest20_resume.log`
  - A separate `/storage/<user>/an_kws/logs/full_mswc_edgespot_mfcc_4_combo_phase1_e5_ep150_manifest20.log` was not created; this is expected for the current resume runner.
- GPU 4 is free after completion:
  - `GPU 4: 11 MiB, 0% util`

## Applied Code Fixes

- `scripts/evaluate.py`
  - Evaluator now resolves checkpoint `frontend_type`/`feature_type`.
  - Fixes DSCNN+PCEN and other ablations being evaluated with the wrong frontend.
- `scripts/evaluate_edgespot_protocol.py`
  - Added `--feature-type` passthrough.
- `src/data/mswc_dataset.py`
  - Removed torchaudio dependency in this path and uses `src.audio_io.load_waveform`.
  - Supports `.opus`, `.wav`, `.ogg`, `.flac`.
  - OPUS is scanned first for current full MSWC server data.
  - Stops scanning a word folder early when `max_per_word` cap is set.
- `scripts/train.py`
  - Added `--n-classes` and `--n-samples` CLI overrides for smoke/matrix runs.
  - Added `--initial-best-metric` so resumed screening runs do not overwrite an earlier better `best.pt` with a worse post-resume checkpoint.
  - `load_checkpoint()` is compatible with both newer PyTorch and ict6 PyTorch 1.12.
- `src/evaluation/metrics.py` and `src/evaluation/protocols.py`
  - Removed `zip(..., strict=True)` so evaluation works on server Python 3.9.
- `server/run_full_mswc_experiment_matrix.sh`
  - Exact 12 requested combinations only.
  - Accepts `MATRIX_N_CLASSES`, `MATRIX_N_SAMPLES`, `MATRIX_MAX_PER_WORD`.
  - Accepts OPUS as supported audio; no WAV conversion required on ict6 because `soundfile` can read OPUS.
- `server/wait_gpu_then_run_full_mswc_matrix.sh`
  - Added `MATRIX_RUNNER` so the same GPU wait logic can launch the resume runner.
- `server/run_full_mswc_experiment_matrix_resume.sh`
  - New script. Skips completed rows, resumes incomplete checkpoints, protects prior `best.pt`, and optionally runs the missing EdgeSpotFull T4 + MFCC supplementary matrix.

## Local Verification Already Run

- `python -m py_compile data/download_mswc.py scripts/evaluate.py scripts/evaluate_edgespot_protocol.py scripts/train.py src/data/mswc_dataset.py src/evaluation/metrics.py src/evaluation/protocols.py`
- `python -m pytest tests/test_evaluate_frontend.py tests/test_dscnn.py tests/test_edgespot_full.py -q` -> 15 passed
- `python -m pytest tests/test_metrics.py tests/test_protocols.py -q` -> 21 passed
- `python -m pytest tests/test_mswc_microset.py tests/test_evaluate_frontend.py -q` -> 7 passed
- `tests/test_mswc_drive_cache.py` currently has unrelated existing failures around legacy Drive cache validation/repair. Do not treat those as matrix blockers unless working specifically on Drive cache.

## Full MSWC Matrix Analysis Artifacts

Generated locally on 2026-05-31 from copied server logs:

- Analysis script:
  - `D:\Downloads\DoAnTotNghiep\scripts\analyze_full_mswc_matrix.py`
- Report directory:
  - `D:\Downloads\DoAnTotNghiep\reports\full_mswc_matrix_analysis`
- Raw evidence copied from server:
  - `reports\full_mswc_matrix_analysis\raw\full_mswc_12_combo_phase1_e5_ep150_manifest20.log`
  - `reports\full_mswc_matrix_analysis\raw\full_mswc_12_combo_phase1_e5_ep150_manifest20_resume.log`
  - `reports\full_mswc_matrix_analysis\raw\full_mswc_12_combo_phase1_e5_ep150_manifest20_runs.tsv`
  - `reports\full_mswc_matrix_analysis\raw\full_mswc_edgespot_mfcc_4_combo_phase1_e5_ep150_manifest20_runs.tsv`
- Generated tables:
  - `matrix_best_epoch_metrics.csv/md`
  - `matrix_effect_deltas.csv/md`
  - `det_curve_summary.csv/md`
- Generated figures:
  - `all_metric_heatmap.png`
  - `research_metric_dashboard.png`
  - `acc1far_ranked_bar.png`
  - `acc1far_interaction_heatmap.png`
  - `loss_effect_lines.png`
  - `key_effect_delta_bars.png`
- Vietnamese interpretation report:
  - `full_mswc_matrix_analysis_vi_notes.md`

Phase-1 interpretation:

- Best overall phase-1 combo by GSC-dev ACC@1%FAR:
  - `DSCNN-L + PCEN + GE2E`: `76.67%` ACC@1%FAR, `79.98%` ACC@5%FAR, `48.97%` FRR@5%FAR, `67.68%` F1.
- Best EdgeSpotFull T4 phase-1 combo:
  - `EdgeSpotFull T4 + PCEN + GE2E`: `72.94%` ACC@1%FAR, `73.35%` ACC@5%FAR, `71.03%` FRR@5%FAR, `57.29%` F1.
- Parameter counts parsed from logs:
  - DSCNN-L + PCEN: `412,900`
  - EdgeSpotFull T4 + PCEN: `130,598`
- Key deltas:
  - `DSCNN-L + GE2E`: PCEN vs MFCC = `+4.37 pp` ACC@1%FAR, `+6.99 pp` F1.
  - `EdgeSpotFull T4 + GE2E`: PCEN vs MFCC = `+3.63 pp` ACC@1%FAR, `+17.91 pp` F1.
  - `DSCNN-L + PCEN`: GE2E vs Triplet = `+4.43 pp` ACC@1%FAR, `+12.70 pp` F1.
  - `EdgeSpotFull T4 + PCEN`: GE2E vs Triplet = `+2.18 pp` ACC@1%FAR, `+12.93 pp` F1.
  - `DSCNN-L + PCEN`: SCAF+GE2E vs GE2E = `-6.56 pp` ACC@1%FAR, `-29.81 pp` F1.
  - `EdgeSpotFull T4 + PCEN`: SCAF+GE2E vs GE2E = `-3.70 pp` ACC@1%FAR, `-19.07 pp` F1.
- Claim hygiene:
  - This matrix is phase-1 GSC-dev screening, not final GSC-test100.
  - It should be used to shortlist longer training/evaluation, not to override Microset evidence without a matched final protocol.
  - `det_curve_summary` is an operating-point DET summary from available logs; full threshold-by-threshold DET curves require saved raw scores.

## Local Cleanup Already Performed

## Active Shortlist Training/Evaluation

Launched on ict6 on 2026-05-31 10:31 ICT:

- tmux session:
  - `kws_shortlist_manifest20`
- Launcher:
  - `/storage/<user>/an_kws/DoAnTotNghiep/server/launch_shortlist_manifest20.sh`
- Runner:
  - `/storage/<user>/an_kws/DoAnTotNghiep/server/run_full_mswc_shortlist_manifest20.sh`
- Wait log:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200_wait_gpu.log`
- Run log:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200.log`
- Summary TSV:
  - `/storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200_runs.tsv`
- Results directory:
  - `/storage/<user>/an_kws/DoAnTotNghiep/results/full_mswc_shortlist_manifest20_e20_ep200`

Shortlist configurations:

- `DSCNN-L + PCEN + GE2E`
  - run tag: `dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200`
- `EdgeSpotFull T4 + PCEN + GE2E`
  - run tag: `edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200`

Settings:

- `RUN_EPOCHS=20`
- `RUN_EPISODES=200`
- `RUN_MAX_PER_WORD=20`
- `RUN_N_CLASSES=30`
- `RUN_N_SAMPLES=10`
- `RUN_WORKERS=8`
- checkpoint selection: GSC-dev every 2 epochs, 5 runs, 10-shot, `TARGET_FAR=0.01`
- final evaluation after each successful train:
  - GSC-dev 30 runs with DET curve
  - GSC-test 100 runs with DET curve

Verified launch evidence:

- GPU 4 passed idle check 2/2 at `2026-05-31 10:34:08` and launched.
- Runtime evidence:
  - `torch: 1.12.1+cu102 cuda: 10.2 available: True`
  - `gpu0: Tesla K80`
- First active run:
  - `dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200`
  - loaded `527069 train file paths`, `10637 val file paths`
  - `DataLoader: 30 classes x 10 samples x 200 episodes`

Latest observed status on 2026-05-31 20:59 ICT:

- tmux session `kws_shortlist_manifest20` still exists.
- First run completed:
  - `dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200`
  - summary row status: `train_status=ok`, `dev30_status=ok`, `test100_status=ok`
  - started: `2026-05-31 10:34:11`
  - finished including dev/test eval: `2026-05-31 14:54:04`
- First run final evaluation artifacts exist:
  - `results/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_k10_results.json`
  - `results/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_det_curve.png`
  - `results/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/test100/gsc_edgespot_exact_k10_results.json`
  - `results/full_mswc_shortlist_manifest20_e20_ep200/dscnn_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/test100/gsc_edgespot_exact_det_curve.png`
- First run metrics at `target_far=0.01`:
  - dev30: open-set ACC@1%FAR `0.8269 ± 0.0126`, FRR `0.5226 ± 0.0421`, AUC `0.9196 ± 0.0068`, EER `0.1589 ± 0.0103`, keyword ACC `0.8796 ± 0.0113`, F1 `0.7640 ± 0.0139`
  - test100: open-set ACC@1%FAR `0.8210 ± 0.0087`, FRR `0.5548 ± 0.0294`, AUC `0.9157 ± 0.0058`, EER `0.1625 ± 0.0086`, keyword ACC `0.8890 ± 0.0125`, F1 `0.7590 ± 0.0116`
- Second run training completed and final evaluation is in progress:
  - `edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200`
  - no active `scripts/train.py` process was observed.
  - active `scripts/evaluate.py` process was observed for test100:
    - checkpoint: `checkpoints/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/best.pt`
    - split: `test`
    - `--n-runs 100`
    - `--target-far 0.01`
  - dev30 completed:
    - output JSON: `results/full_mswc_shortlist_manifest20_e20_ep200/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_k10_results.json`
    - DET curve: `results/full_mswc_shortlist_manifest20_e20_ep200/edgespot_full_t4_pcen_ge2e_full_mswc_shortlist_manifest20_e20_ep200/dev30/gsc_edgespot_exact_det_curve.png`
    - open-set ACC@1%FAR `0.7935 ± 0.0100`
    - FRR@1%FAR `0.6275 ± 0.0349`
    - AUC `0.8855 ± 0.0073`
    - EER `0.1890 ± 0.0103`
    - keyword ACC `0.8275 ± 0.0121`
    - F1 `0.7240 ± 0.0134`
  - test100 was active at run `45/100` in the log.
  - GPU 4 was actively used by evaluation: `6541 MiB`, `100%` utilization.
  - Summary TSV still had only the DSCNN row because the runner writes the EdgeSpot row after test100 finishes.

Server compatibility patch:

- `scripts/evaluate.py` now falls back to plain `torch.load(..., map_location=device)` when ict6 PyTorch 1.12 rejects `weights_only=False`.
- Server verification completed:
  - `bash -n server/run_full_mswc_shortlist_manifest20.sh`
  - `bash -n server/launch_shortlist_manifest20.sh`
  - `python -m py_compile scripts/evaluate.py`
  - `python scripts/evaluate.py --help` shows `--target-far`, `--plot-det`, and `--gsc-query-split`.

Check progress with:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux attach -t kws_shortlist_manifest20
tail -n 80 /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200.log
cat /storage/<user>/an_kws/logs/full_mswc_shortlist_manifest20_e20_ep200_runs.tsv
```

- Removed `.pytest_cache`.
- Removed all `__pycache__` and remaining `.pyc/.pyo` files under the workspace.
- Did not delete checkpoints, data, notebooks, server artifacts, results, or reports.

## Lightweight Server Check

Use lightweight log reads rather than raw `tail` on tqdm-heavy logs:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux ls | grep kws
ps -u <user> -o pid,stat,etime,%cpu,%mem,cmd | grep -E 'scripts/train.py|run_matrix12|run_full_mswc' | grep -v grep
python3 - <<'PY'
from pathlib import Path
for name in [
    "full_mswc_12_combo_manifest20_smoke.log",
    "full_mswc_12_combo_manifest20_smoke_runs.tsv",
]:
    p = Path("/storage/<user>/an_kws/logs") / name
    print("---", name, p.exists())
    if p.exists():
        data = p.read_bytes()[-24000:].decode("utf-8", "replace").replace("\r", "\n")
        for line in [x for x in data.splitlines() if x.strip()][-60:]:
            print(line[:320])
PY
```

## Default Next-Step Discipline

For future non-trivial tasks:

1. Use the 6-role protocol from `docs/agent_orchestration_init.md`.
2. Keep Main/Supervisor on the critical path.
3. Spawn/use specialist agents only for independent bounded work.
4. Persist updated state here after server/job/result changes.

## 2026-06-02 Top500Full Recheck Status

Checked at `2026-06-02 22:58 ICT` from frontend.

Current Top500Full recheck run:

- Log: `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200.log`
- Summary TSV: `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_runs.tsv`
- Data profile: `data/mswc_top500_full`
- Train manifest: `3,346,271` files across `450` train words
- Val manifest: `96,493` files across `50` val words

Completed:

- `evaluate_existing edgespot_full_t4_pcen_scaf_ge2e_top500_epoch13_reval`
  - checkpoint: `checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt`
  - train status: `skipped`
  - dev status: `ok`
  - test status: `ok`
  - meaning: local old Top500 epoch13 checkpoint was uploaded/re-evaluated, not retrained.
- `train_and_eval dscnn_pcen_ge2e_top500_full_recheck_e20_ep200`
  - checkpoint: `checkpoints/dscnn_pcen_ge2e_top500_full_recheck_e20_ep200/best.pt`
  - train status: `ok`
  - dev status: `ok`
  - test status: `ok`
  - final test100 metrics seen in log at target FAR 5%:
    - AUC `0.9317 +/- 0.0050`
    - EER `0.1400 +/- 0.0085`
    - open-set ACC `0.8656 +/- 0.0071`
    - F1 `0.7897 +/- 0.0118`

In progress / issue:

- `edgespot_full_t4_pcen_scaf_ge2e_top500_full_recheck_e20_ep200`
  - started training at `2026-06-02 19:46:27`
  - reached `Epoch 11/20` at `2026-06-02 21:04:55`
  - log file last modified: `2026-06-02 21:04`
  - checkpoint directory contains `epoch_01.pt` through `epoch_10.pt`, plus `best.pt` and `latest.pt`
  - no `epoch_11.pt` and no summary TSV row yet
  - frontend can read `/storage`, but `ssh ict6` timed out during banner exchange at the check time.
  - likely state: job/server is stuck or ict6 is temporarily inaccessible; cannot confirm active process from frontend alone.

Latest EdgeSpot train-new best before stall:

- Best checkpoint after epoch 10: `best.pt`
- GSC-dev at epoch 10:
  - ACC@1%FAR `0.7791`
  - ACC@5%FAR `0.8076`
  - FRR@5% `0.4578`

Recommended next action:

1. Recheck `ssh ict6` availability.
2. If ict6 is reachable, inspect `tmux attach -t kws_top500_recheck` and `ps` for the training PID.
3. If no active training process or it is stuck, resume/restart EdgeSpot Top500Full from `latest.pt`/`best.pt` if train script supports resume; otherwise rerun only the EdgeSpot train-new branch and keep completed re-eval/DSCNN rows.

## 2026-06-03 Top500Full Recheck Follow-Up

Checked at `2026-06-03 11:28 ICT`.

Evidence:

- Frontend reachable: `frontend`, date `Wed Jun 3 11:28:34 +07 2026`.
- `ssh ict6` still fails with `Connection timed out during banner exchange`.
- `/storage` is readable from frontend.
- `top500_full_recheck_e20_ep200.log` still last modified at `2026-06-02 21:04`.
- Summary TSV still has only two rows:
  - legacy EdgeSpot epoch13 re-evaluate: `skipped/ok/ok`
  - DSCNN PCEN GE2E train/eval: `ok/ok/ok`
- EdgeSpot train-new log still stops at `Epoch 11/20` after saving `epoch_10.pt/latest.pt/best.pt`.

Conclusion:

- EdgeSpotFull T4 + PCEN + SCAF+GE2E Top500Full train-new did not finish.
- No final dev/test result exists for that train-new run.
- Last reliable checkpoint is epoch 10 / `latest.pt`.
- Last GSC-dev checkpoint-selection metric before the stall: ACC@1%FAR `0.7791`, ACC@5%FAR `0.8076`.

## 2026-06-03 ict14 Check

Checked at `2026-06-03 14:03 ICT`.

Evidence:

- `ssh ict14` from frontend succeeds.
- `/storage/<user>/an_kws/DoAnTotNghiep` exists on `ict14`.
- Resume checkpoint is visible:
  - `/storage/<user>/an_kws/DoAnTotNghiep/checkpoints/edgespot_full_t4_pcen_scaf_ge2e_top500_full_recheck_e20_ep200/latest.pt`
- `nvidia-smi` is not found on `ict14`.
- No `/dev/nvidia*` devices were visible.
- No `kws_cu102` conda env was found under the checked user env paths.
- System Python is `/usr/bin/python3`, version `3.13.5`, which is not the ict6 CUDA/PyTorch training env.

Conclusion:

- `ict14` is reachable but is not currently suitable for GPU resume of this KWS training job.
- Do not resume the Top500Full train on `ict14` unless a proper CUDA/PyTorch env and GPU visibility are configured.

## 2026-06-04 ict6 Recovered And Resume Launched

Checked at `2026-06-04 09:36-09:42 ICT`.

Evidence:

- `ssh ict6` works again.
- Host: `ictserver6`.
- Original Top500Full recheck log updated after ict6 recovered:
  - `top500_full_recheck_e20_ep200.log` modified at `2026-06-04 08:48`.
  - original runner saved `epoch_11.pt` and updated `latest.pt`, then failed.
- Failure cause from original runner:
  - `OSError: [Errno 12] Cannot allocate memory`
  - occurred when DataLoader tried to fork workers.
  - original run used `RUN_WORKERS=8`.
- Original summary TSV now includes EdgeSpot train-new row:
  - `train_status=failed_rc_1`
  - `dev_status=not_started`
  - `test_status=not_started`
  - `finished_at=2026-06-04 08:48:53`

Action taken:

- Added local/server scripts:
  - `server/resume_top500_edgespot_scaf_ge2e.sh`
  - `server/launch_resume_top500_edgespot.sh`
- Copied both scripts to `/storage/<user>/an_kws/DoAnTotNghiep/server/`.
- Launched tmux session on ict6:
  - `kws_top500_edgespot_resume`
- Resume settings:
  - `RUN_WORKERS=0`
  - `GPU_ID=4`
  - resume checkpoint: `checkpoints/edgespot_full_t4_pcen_scaf_ge2e_top500_full_recheck_e20_ep200/latest.pt`
  - best checkpoint: `checkpoints/edgespot_full_t4_pcen_scaf_ge2e_top500_full_recheck_e20_ep200/best.pt`
  - initial best metric: `0.7791`
- Resume log:
  - `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_edgespot_resume.log`
- Resume summary:
  - `/storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_edgespot_resume_runs.tsv`

Confirmed resume startup:

- `torch: 1.12.1+cu102`
- `cuda: 10.2`
- `available: True`
- `gpu0: Tesla K80`
- loaded `3346271` train file paths and `96493` val file paths.

Follow-up note:

- Subsequent rapid SSH checks to the gateway returned transient `Permission denied`.
- Recheck later with:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux attach -t kws_top500_edgespot_resume
tail -n 80 /storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_edgespot_resume.log
cat /storage/<user>/an_kws/logs/top500_full_recheck_e20_ep200_edgespot_resume_runs.tsv
```

## 2026-07-11 Local Production Demo

The DSCNN production-demo optimization and final verification are documented in:

- `docs/session_handoff_2026_07_11_production_demo.md`

Current verified default profile: DSCNN-L + PCEN + GE2E composite-300. The second featured profile is EdgeSpotFull T4 + PCEN + GE2E composite-300. Older Top500, Microset, legacy, and auto-discovered checkpoints are available only under the expanded model list.

Verified local model results and runtime:

- DSCNN composite: test100 ACC@1%FAR `86.36%`, `412,900` encoder parameters, median single request `26.97 ms` on local CPU.
- EdgeSpot T4 composite: test100 ACC@1%FAR `82.87%`, `130,598` encoder parameters, median single request `21.94 ms` on local CPU.
- Both checkpoints load with the PCEN frontend and `(1, 40, 101)` input.
- Full Python suite: `165 passed` with one torchaudio future warning.
- UI typecheck/build passed. Desktop/mobile checks show exactly two flagship cards by default, no horizontal overflow, and no browser console warnings/errors.
- Local server URL: `http://127.0.0.1:8000/`.

## 2026-07-13 Defense Q&A And Five-Day Study Plan

Created and verified:

- `docs/presentation/defense_practical_qa_and_5_day_plan_vi.md`
- 71 answered defense questions, 30 rapid follow-ups, four detailed English answers, and a five-day plan at six hours/day.
- UTF-8 validation passed with no replacement characters; Q1-Q71 are complete with no duplicate or missing numbers.

Critical implementation facts for defense:

- Encoder is the network; an embedding is one clip's encoder output. Training updates the encoder through embedding losses, while enrollment freezes the encoder and averages support embeddings into a prototype.
- Canonical GSC evaluation normalizes each support embedding, averages 10 supports, and does not normalize the mean again. The robust demo normalizes its mean prototype and uses a mixed prototype/exemplar score. Keep these paths distinct.
- Benchmark `ACC@1%FAR` uses `score = -minimum L2 distance` and selects the ROC operating point whose empirical FAR is at most 1%. This is not a deployment threshold frozen from dev; deployment needs separate calibration and an independently tested frozen threshold.
- DSCNN-MFCC has `412,896` encoder parameters. DSCNN-PCEN has `412,900`: PCEN adds four shared scalars (`alpha`, `delta`, `r`, `s`) because `per_channel=False`. GE2E adds two training-only scalars, not included in the reported encoder count.
- MFCC and mel-PCEN use different input geometry (`1x47x10` versus `1x40x101`), so the comparison is between complete frontend pipelines, not an isolated compression-only ablation. Similar parameter counts do not imply equal MACs or latency.
- Main result claims remain DSCNN-L+PCEN+GE2E `86.36 +/- 1.29%` and EdgeSpotFull T4+PCEN+GE2E `82.87 +/- 1.22%` at GSC test100 ACC@1%FAR.
- MSWC-to-GSC is cross-corpus, not guaranteed lexical-disjoint: six GSC targets were found in the reconstructed cap620 MSWC train vocabulary. Do not call the result strict unseen-vocabulary evaluation.

Practical readiness target:

- Thirty focused hours are sufficient for defense-level mastery of the critical path, not line-by-line memorization of the entire repository.
- Memorize six anchors: `10-shot`, `100 runs`, `1% FAR`, DSCNN `86.36% / 412,900`, EdgeSpot `82.87% / 130,598`.
- Answer in four parts: problem, decision, evidence, limitation.

## 2026-07-15 Consolidated 30-Hour Mastery Curriculum

Created the single comprehensive study artifact requested for code, slides,
script, exercises, and defense preparation:

- `docs/learning/kws_30_hour_code_slide_mastery_vi.md`

Scope and structure:

- 5 days x 6 hours with all 30 hourly sections present;
- source triage across 108 Python files into deep-read, call-flow, and awareness
  levels;
- exact training, canonical test100, and robust-demo call flows;
- runnable PowerShell/Python exercises with expected shapes/results;
- handwritten calculations for prototype/L2/FAR/FRR/ACC/AUC/EER/F1;
- result JSON/checkpoint integrity exercises and claim boundaries;
- full 17-slide-to-source crosswalk, script rehearsal, 30 oral questions,
  flashcards, progress checklist, and defense-day schedule;
- current slide/script claim corrections, especially three development branches,
  per-run ROC threshold semantics, cross-corpus-not-lexical-zero-shot wording,
  and compactness versus target-device efficiency.

Verification completed:

- `scripts/trace_core_pipeline.py` reproduced all expected shapes and parameter
  counts (`412,896`, `412,900`, `130,598`).
- The consolidated focused Python suite covering frontend, encoders, losses,
  evaluation, profiles, API, and robust streaming passed (exit code 0).
- `npm.cmd run typecheck` passed in `src/demo/ui`.
- Markdown audit: 2,187 lines, exactly 30 hourly sections, 5 day sections,
  balanced code fences, UTF-8 content, and all referenced focused test files
  exist.

No model, checkpoint, result, slide PDF, or server process was changed.
## 2026-07-17 Hostile-Jury Defense Audit

- Added `docs/presentation/defense_hostile_jury_title_formula_pipeline_streaming_qa_vi.md` with 72 evidence-scoped questions, rapid follow-ups, figure/slide traps, and a show-code map.
- Critical title mismatch: the user confirmed the registered title is `Enhanced Few-Shot Open-Set Keyword Spotting with Noise-Robust Prototype Classification and Real-Time Streaming` (without `Inference`). Proposal/outline, compiled English thesis, and slide 1 use different wording. The slide/script should be corrected immediately, and the submitted thesis mismatch must be reported to the supervisor/academic office for an approved replacement page, resubmission, or erratum.
- The official-topic wording is supported only with scoped claims: PCEN plus DEMAND/gain augmentation is noise-aware, but there is no controlled SNR/denoising test; streaming is implemented and locally benchmarked, but the thesis reports design budgets and lacks field FA/hour/onset-to-detection validation.
- Thesis streaming text describes the fixed 1 s / 0.5 s baseline in `src/streaming/vad_engine.py`; the current API uses the richer rolling-buffer `RobustStreamingKWS` path in `src/streaming/robust_engine.py`. Keep these implementations distinct when answering.

## 2026-07-17 DEMAND And Streaming Evidence Correction

- `src/features/augmentation.py` implements RMS-scaled DEMAND noise mixing and
  supports random SNR in `0-10 dB`; `tests/test_augmentation.py` unit-tests the
  behavior.
- `scripts/train.py` enables the augmenter only if `data/demand` exists.
- Audit evidence: `data/demand` is missing both locally and under the ICTLab
  project; the composite-300 runner does not download DEMAND; retained final
  checkpoints do not store augmentation config; retained logs contain no active
  `Noise augmentation:` line.
- Therefore do not claim the two headline composite checkpoints were verified to
  use DEMAND. Treat it as an implemented/tested optional path and mark final-run
  activation as a reproducibility gap.
- Live streaming is implemented via `/ws/stream`, a 3.5 s rolling buffer, 250 ms
  cadence, multi-duration windows, threshold/margin/votes, and cooldown. The raw
  engineering RTF artifact uses offline `process_file()`, so it proves compute
  throughput only, not live onset-to-detection latency or field FA/hour.
- Updated the reviewed script, the 30-hour curriculum, and hostile-jury Q&A.
- Added `docs/presentation/slide_noise_streaming_corrections_2026_07_17.md` with
  exact slide copy and claim boundaries.

## 2026-07-17 Final Presentation Title Decision

- The submitted thesis title is the presentation source of truth:
  `Few-Shot Open-Set Keyword Spotting at Vocabulary Scale`, with subtitle
  `A Metric-Learning Study of Feature Front-Ends, Encoders, and Open-Set Rejection`.
- Slide 1 and the reviewed oral script should use that exact thesis title/subtitle.
- The April registered title is described as the broader intended scope. By July,
  controlled noise and field-streaming evidence remained incomplete, so the thesis
  headline was narrowed to the fully evaluated vocabulary-scale metric-learning work.
- The defense answer must acknowledge the administrative synchronization mistake and
  must not claim that an official title change was approved without evidence.

## 2026-07-17 Slides 7-16 Copy Audit

- Added `docs/presentation/slides_7_16_exact_copy_2026_07_17.md` with exact concise
  slide text aligned to the reviewed eight-minute script.
- Corrected the spoken main-result description from two to three extended-training
  branches; the slide still reports the two successful final profiles.
- Kept slide 15 as a general demo slide and scoped streaming as an engineering
  prototype rather than a validated field result.
- Corrected slide 10 architecture labels: PCEN receives non-log mel energy, encoder
  boxes report raw output dimensions, and L2 normalization is external to the model.
