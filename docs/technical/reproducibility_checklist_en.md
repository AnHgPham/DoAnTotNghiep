# Reproducibility Checklist EN

## Before Training

- [ ] Confirm branch and code version.
- [ ] Confirm YAML config.
- [ ] Confirm dataset path.
- [ ] For Microset, use official CSV manifests.
- [ ] For Top500, confirm 450 train and 50 validation words.
- [ ] Confirm DEMAND if noise augmentation is enabled.
- [ ] On Colab, checkpoint output must point to Drive.

## During Training

- [ ] Save the exact command.
- [ ] For Top500, use `--save-every 1`.
- [ ] For Top500, use `--save-latest-every-epoch`.
- [ ] Monitor loss, GE2E accuracy, validation AUC/ACC.
- [ ] Monitor GSC-dev if checkpoint selection is enabled.
- [ ] Reduce DataLoader workers if the runtime freezes.

## After Training

- [ ] Verify `epoch_XX.pt`.
- [ ] Verify `latest.pt`.
- [ ] Verify `best.pt` when applicable.
- [ ] Run dev30 or test100 evaluation.
- [ ] Save result JSON.
- [ ] Save DET curve.
- [ ] Package artifacts to Drive before relying on local download.

## Local Artifact

- [ ] Copy checkpoint into `server`.
- [ ] Copy result JSON into `server`.
- [ ] Copy DET curve.
- [ ] Run `python scripts/make_project_status.py`.
- [ ] Inspect `reports/project_status/claim_matrix.md`.
- [ ] Only claim results with evidence.

## Demo

- [ ] Start FastAPI.
- [ ] Build React UI with `npm run build`.
- [ ] Confirm model cards are ready.
- [ ] Enroll GSC 17 known.
- [ ] Run single detection.
- [ ] Run long audio with labels/timings.
- [ ] Run open-set 17/17.
- [ ] Run calibration.
- [ ] Export session report.

## Thesis/Report

- [ ] Treat Microset as the main thesis anchor.
- [ ] Treat Top500 epoch13 as local preliminary/demo evidence.
- [ ] Mention Top500 epoch25 only as historical progress if no complete artifact exists.
- [ ] Label Open-set UI as sampled demo-level evaluation.
- [ ] Do not claim full EdgeSpot paper reproduction.
